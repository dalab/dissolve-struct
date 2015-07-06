package ch.ethz.dalab.dissolve.examples.imageseg

import scala.annotation.elidable
import scala.annotation.elidable.ASSERTION
import scala.collection.mutable.ArrayBuffer
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.linalg.argmax
import cc.factorie.infer.MaximizeByMPLP
import cc.factorie.model.Factor
import cc.factorie.model.ItemizedModel
import cc.factorie.singleFactorIterable
import cc.factorie.variable.DiscreteDomain
import cc.factorie.variable.DiscreteVariable
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegmentationTypes.AdjacencyList
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegmentationTypes.Label
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegmentationTypes.RGB_INT
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import cc.factorie.model.Factor2
import cc.factorie.model.Factor1

/**
 * `data` is a `F` x `N` matrix; each column contains the features for a super-pixel
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
                          rgbArray: Array[RGB_INT] = null)
/**
 * labels(i) contains label for `i+1`th super-pixel
 */
case class QuantizedLabel(labels: Array[Int],
                          filename: String = "NA")

/**
 * Functions for dissolve^struct
 */
object ImageSegmentationAdv
    extends DissolveFunctions[QuantizedImage, QuantizedLabel] {

  val NUM_CLASSES: Int = 22 // # Classes (0-indexed)
  val BACKGROUND_CLASS: Int = 21 // The last label

  val INTENSITY_LEVELS: Int = 4
  val NUM_BINS = INTENSITY_LEVELS * INTENSITY_LEVELS * INTENSITY_LEVELS // Size of feature vector x_i

  val DISABLE_PAIRWISE = true

  /**
   * Joint Feature Map
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

    val unaryFeatures = DenseMatrix.zeros[Double](d, NUM_CLASSES)
    for (superIdx <- 0 until numSuperpixels) {
      val thisLabel = y.labels(superIdx)

      val x_i = x.unaryFeatures(::, superIdx)
      unaryFeatures(::, thisLabel) := unaryFeatures(::, thisLabel) + x_i
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

      DenseVector.vertcat(unaryFeatures.toDenseVector, transitions.toDenseVector)
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
   * Structured Error Function
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
        def score(k: Pixel#Value) = unaryPot(k.intValue, idx)
      }
    }

    def getPairwiseFactor(yi: Pixel, yj: Pixel): Factor = {
      new Factor2(yi, yj) {
        def score(i: Pixel#Value, j: Pixel#Value) = pairwisePot(i.intValue, j.intValue)
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

    // MaximizeByBPLoopy.maximize(pixelSeq, model)
    val maxIterations = 100
    val maximizer = new MaximizeByMPLP(maxIterations)
    val assgn = maximizer.infer(pixelSeq, model).mapAssignment

    val mapLabels: Array[Label] = (0 until nSuperpixels).map {
      idx =>
        assgn(pixelSeq(idx)).intValue
    }.toArray

    mapLabels
  }

  /**
   * Unpack weight vector to Unary and Pairwise weights
   */
  def unpackWeightVec(weightv: DenseVector[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {

    assert(weightv.size >= (NUM_CLASSES * NUM_BINS))
    val d = NUM_BINS

    val unaryWeights = weightv(0 until NUM_CLASSES * d)
    val unaryWeightMat = unaryWeights.toDenseMatrix.reshape(d, NUM_CLASSES)

    val pairwisePot =
      if (!DISABLE_PAIRWISE) {
        assert(weightv.size == (NUM_CLASSES * NUM_BINS) + (NUM_CLASSES * NUM_CLASSES))
        val pairwiseWeights = weightv((NUM_CLASSES * d) until weightv.size)
        pairwiseWeights.toDenseMatrix.reshape(NUM_CLASSES, NUM_CLASSES)
      } else null

    (unaryWeightMat, pairwisePot)
  }

  /**
   * Maximization Oracle
   */
  override def oracleFn(model: StructSVMModel[QuantizedImage, QuantizedLabel],
                        xi: QuantizedImage,
                        yi: QuantizedLabel): QuantizedLabel = {

    val nSuperpixels = xi.unaries.cols
    val d = xi.unaryFeatures.rows

    assert(xi.pairwise.length == nSuperpixels,
      "xi.pairwise.length == nSuperpixels")
    assert(xi.unaryFeatures.cols == nSuperpixels,
      "xi.pairwise.length == nSuperpixels")

    val (unaryWeights, pairwisePot) = unpackWeightVec(model.weights.toDenseVector)
    val unaryPot = unaryWeights.t * xi.unaryFeatures

    if (yi != null) {
      assert(yi.labels.length == xi.pairwise.length,
        "yi.labels.length == xi.pairwise.length")

      // Loss augment the scores
      for (superIdx <- 0 until nSuperpixels) {
        val trueLabel = yi.labels(superIdx)

        for (label <- 0 until NUM_CLASSES) {
          unaryPot(label, superIdx) = unaryPot(label, superIdx) + perLabelLoss(trueLabel, label)
        }

      }
    }

    val decodedLabels = decode(unaryPot, pairwisePot, xi.pairwise)

    QuantizedLabel(decodedLabels, xi.filename)
  }

  def predictFn(model: StructSVMModel[QuantizedImage, QuantizedLabel],
                xi: QuantizedImage): QuantizedLabel = {
    oracleFn(model, xi, null)
  }

}