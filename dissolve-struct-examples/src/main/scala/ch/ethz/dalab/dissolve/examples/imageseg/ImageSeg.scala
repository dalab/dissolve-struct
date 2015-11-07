package ch.ethz.dalab.dissolve.examples.imageseg

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.linalg.normalize
import cc.factorie.infer.MaximizeByBPLoopy
import cc.factorie.la.DenseTensor1
import cc.factorie.la.Tensor
import cc.factorie.model._
import cc.factorie.singleFactorIterable
import cc.factorie.variable.DiscreteDomain
import cc.factorie.variable.DiscreteVariable
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegTypes.AdjacencyList
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegTypes.Label
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegTypes.RGB_INT
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions

/**
 * `data` is a `d` x `N` matrix; each column contains the features of super-pixel
 * `transitions` is an adjacency list. transitions(i) contains neighbours of super-pixel `i+1`'th super-pixel
 * `pixelMapping` contains mapping of super-pixel to corresponding pixel in the original image (column-major)
 */
case class QuantizedImage(unaries: DenseMatrix[Double],
                          pairwise: Array[Array[Int]],
                          pixelMapping: Array[Int],
                          width: Int,
                          height: Int,
                          filename: String = "NA",
                          unaryFeatures: DenseMatrix[Double] = null,
                          rgbArray: Array[RGB_INT] = null,
                          globalFeatures: Vector[Double] = null)

/**
 * labels(i) contains label for `i`th super-pixel
 */
case class QuantizedLabel(labels: Array[Int],
                          filename: String = "NA")

/**
 * Functions for dissolve^struct
 *
 * Designed for the MSRC-21 dataset
 */
object ImageSeg
    extends DissolveFunctions[QuantizedImage, QuantizedLabel] {

  val NUM_CLASSES: Int = 22 // # Classes (0-indexed)
  val BACKGROUND_CLASS: Int = 21 // The last label

  val INTENSITY_LEVELS: Int = 8
  val NUM_BINS = INTENSITY_LEVELS * INTENSITY_LEVELS * INTENSITY_LEVELS // Size of feature vector x_i

  var DISABLE_PAIRWISE = false

  /**
   * ======= Joint Feature Map =======
   */
  def featureFn(x: QuantizedImage, y: QuantizedLabel): Vector[Double] = {

    val numSuperpixels = x.unaries.cols // # Super-pixels
    val classifierScore = x.unaries.rows // Score of SP per label

    assert(numSuperpixels == x.pairwise.length,
      "numSuperpixels == x.pairwise.length")

    /**
     * Unary Features
     */
    val d = x.unaryFeatures.rows
    assert(d == NUM_BINS, "d == NUM_BINS")

    val dCombined =
      if (x.globalFeatures != null)
        d + x.globalFeatures.size
      else
        d

    val unaryFeatures = DenseMatrix.zeros[Double](dCombined, NUM_CLASSES)
    for (superIdx <- 0 until numSuperpixels) {
      val x_i = x.unaryFeatures(::, superIdx)
      val x_global = x.globalFeatures

      val x_comb =
        if (x_global == null)
          x_i
        else
          Vector(Array.concat(x_i.toArray, x_global.toArray))
      val label = y.labels(superIdx)
      unaryFeatures(::, label) += x_comb
    }

    if (DISABLE_PAIRWISE)
      unaryFeatures.toDenseVector
    else {
      /**
       * Pairwise features
       */
      val transitions = DenseMatrix.zeros[Double](NUM_CLASSES, NUM_CLASSES)
      for (superIdx <- 0 until numSuperpixels) {
        val thisLabel = y.labels(superIdx)

        x.pairwise(superIdx).foreach {
          case adjacentSuperIdx =>
            val nextLabel = y.labels(adjacentSuperIdx)

            transitions(thisLabel, nextLabel) += 1.0
            transitions(nextLabel, thisLabel) += 1.0
        }
      }
      DenseVector.vertcat(unaryFeatures.toDenseVector,
        normalize(transitions.toDenseVector, 2))
    }
  }

  /**
   * Per-label Hamming loss
   */
  def perLabelLoss(labTruth: Label, labPredict: Label): Double =
    if (labTruth == labPredict)
      0.0
    else
      1.0

  /**
   * ======= Structured Error Function =======
   */
  def lossFn(yTruth: QuantizedLabel, yPredict: QuantizedLabel): Double = {

    assert(yTruth.labels.size == yPredict.labels.size,
      "Failed: yTruth.labels.size == yPredict.labels.size")

    val stuctHammingLoss = yTruth.labels
      .zip(yPredict.labels)
      .map {
        case (labTruth, labPredict) =>
          perLabelLoss(labTruth, labPredict)
      }

    // Return normalized hamming loss
    stuctHammingLoss.sum / stuctHammingLoss.length
  }

  /**
   * Construct Factor graph and run MAP
   * (Max-product using Loopy Belief Propogation)
   */
  def decode(unaryPot: DenseMatrix[Double],
             pairwisePot: DenseMatrix[Double],
             adj: AdjacencyList): Array[Label] = {

    val nSuperpixels = unaryPot.cols
    val nClasses = unaryPot.rows

    assert(nClasses == NUM_CLASSES)
    if (!DISABLE_PAIRWISE)
      assert(pairwisePot.rows == NUM_CLASSES)

    object PixelDomain extends DiscreteDomain(nClasses)

    class Pixel(i: Int) extends DiscreteVariable(i) {
      def domain = PixelDomain
    }

    def getUnaryFactor(yi: Pixel, idx: Int): Factor = {
      new Factor1(yi) {
        val weights: DenseTensor1 = new DenseTensor1(unaryPot(::, idx).toArray)
        def score(k: Pixel#Value) = unaryPot(k.intValue, idx)
        override def valuesScore(tensor: Tensor): Double = {
          weights dot tensor
        }
      }
    }

    def getPairwiseFactor(yi: Pixel, yj: Pixel): Factor = {
      new Factor2(yi, yj) {
        val weights: DenseTensor1 = new DenseTensor1(pairwisePot.toArray)
        def score(i: Pixel#Value, j: Pixel#Value) = pairwisePot(i.intValue, j.intValue)
        override def valuesScore(tensor: Tensor): Double = {
          weights dot tensor
        }
      }
    }

    val pixelSeq: IndexedSeq[Pixel] =
      (0 until nSuperpixels).map(x => new Pixel(12))

    val unaryFactors: IndexedSeq[Factor] =
      (0 until nSuperpixels).map {
        case idx =>
          getUnaryFactor(pixelSeq(idx), idx)
      }

    val model = new ItemizedModel
    model ++= unaryFactors

    if (!DISABLE_PAIRWISE) {
      val pairwiseFactors =
        (0 until nSuperpixels).flatMap {
          case thisIdx =>
            val thisFactors = new ArrayBuffer[Factor]

            adj(thisIdx).foreach {
              case nextIdx =>
                thisFactors ++=
                  getPairwiseFactor(pixelSeq(thisIdx), pixelSeq(nextIdx))
            }
            thisFactors
        }
      model ++= pairwiseFactors
    }

    MaximizeByBPLoopy.maximize(pixelSeq, model)

    val mapLabels: Array[Label] = (0 until nSuperpixels).map {
      idx =>
        pixelSeq(idx).intValue
    }.toArray

    mapLabels
  }

  /**
   * Unpack weight vector to Unary and Pairwise weights
   */

  def unpackWeightVec(weightv: DenseVector[Double], d: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {

    assert(weightv.size >= (NUM_CLASSES * d))

    val unaryWeights = weightv(0 until NUM_CLASSES * d)
    val unaryWeightMat = unaryWeights.toDenseMatrix.reshape(d, NUM_CLASSES)

    val pairwisePot =
      if (!DISABLE_PAIRWISE) {
        assert(weightv.size == (NUM_CLASSES * d) + (NUM_CLASSES * NUM_CLASSES))
        val pairwiseWeights = weightv((NUM_CLASSES * d) until weightv.size)
        pairwiseWeights.toDenseMatrix.reshape(NUM_CLASSES, NUM_CLASSES)
      } else null

    (unaryWeightMat, pairwisePot)
  }

  /**
   * ======= Maximization Oracle =======
   */
  override def oracleFn(model: StructSVMModel[QuantizedImage, QuantizedLabel],
                        xi: QuantizedImage,
                        yi: QuantizedLabel): QuantizedLabel = {

    val nSuperpixels = xi.unaryFeatures.cols
    val d = xi.unaryFeatures.rows
    val dComb =
      if (xi.globalFeatures == null)
        d
      else
        d + xi.globalFeatures.size

    assert(xi.pairwise.length == nSuperpixels,
      "xi.pairwise.length == nSuperpixels")
    assert(xi.unaryFeatures.cols == nSuperpixels,
      "xi.unaryFeatures.cols == nSuperpixels")

    val (unaryWeights, pairwisePot) = unpackWeightVec(model.weights.toDenseVector, dComb)
    val localFeatures =
      if (xi.globalFeatures == null)
        xi.unaryFeatures
      else {
        // Concatenate global features to local features
        // The order is : local || global
        val dGlob = xi.globalFeatures.size
        val glob = xi.globalFeatures.toDenseVector
        val globalFeatures = DenseMatrix.zeros[Double](dGlob, nSuperpixels)
        for (superIdx <- 0 until nSuperpixels) {
          globalFeatures(::, superIdx) := glob
        }
        DenseMatrix.vertcat(xi.unaryFeatures, globalFeatures)
      }
    val unaryPot = unaryWeights.t * localFeatures

    if (yi != null) {
      assert(yi.labels.length == xi.pairwise.length,
        "yi.labels.length == xi.pairwise.length")

      // Loss augment the scores
      for (superIdx <- 0 until nSuperpixels) {
        val trueLabel = yi.labels(superIdx)
        // FIXME Use \delta here
        unaryPot(::, superIdx) += (1.0 / nSuperpixels)
        unaryPot(trueLabel, superIdx) -= (1.0 / nSuperpixels)
      }
    }

    val t0 = System.currentTimeMillis()
    val decodedLabels = decode(unaryPot, pairwisePot, xi.pairwise)
    val oracleSolution = QuantizedLabel(decodedLabels, xi.filename)
    val t1 = System.currentTimeMillis()

    oracleSolution
  }

  def predictFn(model: StructSVMModel[QuantizedImage, QuantizedLabel],
                xi: QuantizedImage): QuantizedLabel = {
    oracleFn(model, xi, null)
  }

}