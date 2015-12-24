package ch.ethz.dalab.dissolve.diagnostics

import org.scalatest.FlatSpec
import org.scalatest.Inside
import org.scalatest.Inspectors
import org.scalatest.Matchers
import org.scalatest.OptionValues

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Matrix
import breeze.linalg.Vector
import breeze.linalg.max
import ch.ethz.dalab.dissolve.classification.MutableWeightsEll
import ch.ethz.dalab.dissolve.models.LinearChainCRF
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.regression.LabeledObject

object ChainTestAdapter {
  type X = Matrix[Double]
  type Y = Vector[Double]

  /**
   * Reads data produced by the convert-ocr-data.py script and loads into memory as a vector of Labeled objects
   *
   */
  def loadData(patternsFilename: String, labelsFilename: String, foldFilename: String): Array[LabeledObject[Matrix[Double], Vector[Double]]] = {
    val patterns: Array[String] = scala.io.Source.fromFile(patternsFilename).getLines().toArray[String]
    val labels: Array[String] = scala.io.Source.fromFile(labelsFilename).getLines().toArray[String]
    val folds: Array[String] = scala.io.Source.fromFile(foldFilename).getLines().toArray[String]

    val n = labels.size

    assert(patterns.size == labels.size, "#Patterns=%d, but #Labels=%d".format(patterns.size, labels.size))
    assert(patterns.size == folds.size, "#Patterns=%d, but #Folds=%d".format(patterns.size, folds.size))

    val data: Array[LabeledObject[Matrix[Double], Vector[Double]]] = Array.fill(n) { null }

    for (i <- 0 until n) {
      // Expected format: id, #rows, #cols, (pixels_i_j,)* pixels_n_m
      val patLine: List[Double] = patterns(i).split(",").map(x => x.toDouble) toList
      // Expected format: id, #letters, (letters_i)* letters_n
      val labLine: List[Double] = labels(i).split(",").map(x => x.toDouble) toList

      val patNumRows: Int = patLine(1) toInt
      val patNumCols: Int = patLine(2) toInt
      val labNumEles: Int = labLine(1) toInt

      assert(patNumCols == labNumEles, "pattern_i.cols == label_i.cols violated in data")

      val patVals: Array[Double] = patLine.slice(3, patLine.size).toArray[Double]
      // The pixel values should be Column-major ordered
      val thisPattern: DenseMatrix[Double] = DenseVector(patVals).toDenseMatrix.reshape(patNumRows, patNumCols)

      val labVals: Array[Double] = labLine.slice(2, labLine.size).toArray[Double]
      assert(List.fromArray(labVals).count(x => x < 0 || x > 26) == 0, "Elements in Labels should be in the range [0, 25]")
      val thisLabel: DenseVector[Double] = DenseVector(labVals)

      assert(thisPattern.cols == thisLabel.size, "pattern_i.cols == label_i.cols violated in Matrix representation")

      data(i) = new LabeledObject(thisLabel, thisPattern)

    }

    data
  }

  /**
   * Dissolve Functions
   */
  val dissolveFunctions: DissolveFunctions[X, Y] = new LinearChainCRF(26)
  /**
   * Some Data
   */
  val data = {
    val dataDir = "../data/generated"
    val trainDataSeq: Array[LabeledObject[Matrix[Double], Vector[Double]]] =
      loadData(dataDir + "/patterns_train.csv",
        dataDir + "/labels_train.csv",
        dataDir + "/folds_train.csv")

    trainDataSeq
  }
  /**
   * A dummy model
   */
  val lo = data(0)
  val numd = dissolveFunctions.featureFn(lo.pattern, lo.label).size
  val model: MutableWeightsEll =
    new MutableWeightsEll(DenseVector.zeros(numd), 0.0)

  /**
   * Perturb
   * Return a compatible perturbed Y
   * Higher the degree, more perturbed y is
   *
   * This function perturbs `degree` of the values by swapping
   */
  def perturb(y: Y, degree: Double = 0.1): Y = {
    val d = y.size
    val numSwaps = max(1, (degree * d).toInt)

    for (swapNo <- 0 until numSwaps) {
      // Swap two random values in y
      val (i, j) = (scala.util.Random.nextInt(d), scala.util.Random.nextInt(d))
      val temp = y(i)
      y(i) = y(j)
      y(j) = temp
    }

    y

  }
}

abstract class UnitSpec extends FlatSpec with Matchers with OptionValues with Inside with Inspectors {

  val DissolveAdapter = ChainTestAdapter

  type X = DissolveAdapter.X
  type Y = DissolveAdapter.Y

  val dissolveFunctions = DissolveAdapter.dissolveFunctions
  val data = DissolveAdapter.data
  val model = DissolveAdapter.model

  /**
   * Helper functions
   */
  def perturb = DissolveAdapter.perturb _

  // Joint Feature Map
  def phi = dissolveFunctions.featureFn _
  def delta = dissolveFunctions.lossFn _
  def maxoracle = dissolveFunctions.oracleFn _
  def predict = dissolveFunctions.predictFn _

  def psi(lo: LabeledObject[X, Y], ymap: Y) =
    phi(lo.pattern, lo.label) - phi(lo.pattern, ymap)

  def F(x: X, y: Y, w: Vector[Double]) =
    w dot phi(x, y)
  def deltaF(lo: LabeledObject[X, Y], ystar: Y, w: Vector[Double]) =
    w dot psi(lo, ystar)

}