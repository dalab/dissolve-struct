package ch.ethz.dalab.dissolve.examples.imageseg

import breeze.linalg._
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegmentationTypes._
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import cc.factorie.variable.DiscreteVariable
import cc.factorie.variable.IntegerVariable
import cc.factorie.variable.DiscreteDomain
import cc.factorie.model.Factor2
import cc.factorie.model.Factor1
import cc.factorie.model.Factor
import scala.collection.mutable.ArrayBuffer
import cc.factorie.model.ItemizedModel
import cc.factorie.infer.MaximizeByMPLP

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
                          filename: String = "NA")
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

  /**
   * Joint Feature Map
   */
  def featureFn(x: QuantizedImage, y: QuantizedLabel): Vector[Double] = {

    val numSuperpixels = x.unaries.cols // # Super-pixels
    val classifierScore = x.unaries.rows // Score of SP per label

    assert(numSuperpixels == x.pairwise.length,
      "numSuperpixels == x.pairwise.length")

    val unaryScores = DenseVector.zeros[Double](NUM_CLASSES * 1)
    val transitions = DenseMatrix.zeros[Double](NUM_CLASSES, NUM_CLASSES)

    /**
     * Unary features
     * FIXME
     */
    for (superpixelIdx <- 0 until numSuperpixels) {
      val label = y.labels(superpixelIdx)
      unaryScores(label) += x.unaries(label, superpixelIdx)
    }

    /**
     * Pairwise features
     */
    for (superIdx <- 0 until numSuperpixels) {
      val thisLabel = y.labels(superIdx)

      x.pairwise(superIdx).foreach {
        case adjacentSuperIdx =>
          val nextLabel = y.labels(adjacentSuperIdx)
          transitions(thisLabel, nextLabel) += 1.0
          transitions(nextLabel, thisLabel) += 1.0
      }
    }

    // DenseVector.vertcat(unaryScores, transitions.toDenseVector)
    transitions.toDenseVector
  }

  /**
   * Per-label Hamming loss
   */
  def perLabelLoss(labTruth: Label, labPredict: Label): Double =
    if (labPredict == BACKGROUND_CLASS)
      1.0
    else if (labTruth == BACKGROUND_CLASS)
      0.0
    else if (labTruth == labPredict)
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
      .filter {
        case (labTruth, labPredict) =>
          labTruth != BACKGROUND_CLASS
      }
      .map {
        case (labTruth, labPredict) =>
          perLabelLoss(labTruth, labPredict)
      }

    // Return normalized hamming loss

    stuctHammingLoss.sum / stuctHammingLoss.length
  }

  def unpackWeightVec(weightv: DenseVector[Double]): (DenseVector[Double], DenseMatrix[Double]) = {
    /*val unaryScores = weightv(0 until NUM_CLASSES)

    val pairwiseVec = weightv(NUM_CLASSES until weightv.size)
    val pairwisePot = pairwiseVec.toDenseMatrix.reshape(NUM_CLASSES, NUM_CLASSES)

    (unaryScores, pairwisePot)*/

    assert(weightv.size == NUM_CLASSES * NUM_CLASSES)
    val pairwisePot = weightv.toDenseMatrix.reshape(NUM_CLASSES, NUM_CLASSES)

    (null, pairwisePot)
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
    assert(pairwisePot.rows == NUM_CLASSES)

    object PixelDomain extends DiscreteDomain(nClasses)

    class Pixel(i: Int) extends DiscreteVariable(i) {
      def domain = PixelDomain
    }

    def getUnaryFactor(yi: Pixel, idx: Int): Factor = {
      new Factor1(yi) {
        def score(k: Pixel#Value) = -unaryPot(k.intValue, idx)
      }
    }

    def getPairwiseFactor(yi: Pixel, yj: Pixel): Factor = {
      new Factor2(yi, yj) {
        def score(i: Pixel#Value, j: Pixel#Value) = -pairwisePot(i.intValue, j.intValue)
      }
    }

    val pixelSeq: IndexedSeq[Pixel] =
      (0 until nSuperpixels).map(x => new Pixel(0))

    val unaryFactors: IndexedSeq[Factor] =
      (0 until nSuperpixels).map {
        case idx =>
          getUnaryFactor(pixelSeq(idx), idx)
      }

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

    val model = new ItemizedModel
    model ++= unaryFactors
    model ++= pairwiseFactors

    val maxIterations = 200
    val maximizer = new MaximizeByMPLP(maxIterations)
    val assgn = maximizer.infer(pixelSeq, model).mapAssignment

    val mapLabels: Array[Label] = (0 until nSuperpixels).map {
      idx =>
        assgn(pixelSeq(idx)).intValue
    }.toArray

    mapLabels
  }

  /**
   * Maximization Oracle
   */
  override def oracleFn(model: StructSVMModel[QuantizedImage, QuantizedLabel],
                        xi: QuantizedImage,
                        yi: QuantizedLabel): QuantizedLabel = {

    println("Decoding: " + xi.filename)

    val nSuperpixels = xi.unaries.cols

    assert(xi.pairwise.length == nSuperpixels,
      "xi.pairwise.length == nSuperpixels")

    val (unaryScores, pairwisePot) = unpackWeightVec(model.weights.toDenseVector)
    val unaryPot = xi.unaries

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