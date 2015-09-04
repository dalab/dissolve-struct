package ch.ethz.dalab.dissolve.diagnostics

import java.nio.file.Paths
import org.scalatest.FlatSpec
import org.scalatest.Inside
import org.scalatest.Inspectors
import org.scalatest.Matchers
import org.scalatest.OptionValues
import breeze.linalg.DenseVector
import breeze.linalg.Matrix
import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.examples.chain.ChainDemo
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSeg
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegUtils
import ch.ethz.dalab.dissolve.examples.imageseg.QuantizedImage
import ch.ethz.dalab.dissolve.examples.imageseg.QuantizedLabel
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg.max

object ImageTestAdapter {
  type X = QuantizedImage
  type Y = QuantizedLabel

  /**
   * Dissolve Functions
   */
  val dissolveFunctions: DissolveFunctions[X, Y] = ImageSeg
  /**
   * Some Data
   */
  val data = {
    val dataDir = "../data/generated/msrc"
    val trainFilePath = Paths.get(dataDir, "Train.txt")
    val trainDataSeq = ImageSegUtils.loadData(dataDir, trainFilePath, limit = 50)

    trainDataSeq
  }
  /**
   * A dummy model
   */
  val lo = data(0)
  val numd = ImageSeg.featureFn(lo.pattern, lo.label).size
  val model: StructSVMModel[X, Y] =
    new StructSVMModel[X, Y](DenseVector.zeros(numd), 0.0,
      DenseVector.zeros(numd), dissolveFunctions, 1)

  def perturb(y: Y, degree: Double = 0.1): Y = {
    val d = y.labels.size
    val numSwaps = max(1, (degree * d).toInt)

    for (swapNo <- 0 until numSwaps) {
      // Swap two random values in y
      val (i, j) = (scala.util.Random.nextInt(d), scala.util.Random.nextInt(d))
      val temp = y.labels(i)
      y.labels(i) = y.labels(j)
      y.labels(j) = temp
    }

    y

  }
}

object ChainTestAdapter {
  type X = Matrix[Double]
  type Y = Vector[Double]

  /**
   * Dissolve Functions
   */
  val dissolveFunctions: DissolveFunctions[X, Y] = ChainDemo
  /**
   * Some Data
   */
  val data = {
    val dataDir = "../data/generated"
    val trainDataSeq: Vector[LabeledObject[Matrix[Double], Vector[Double]]] =
      ChainDemo.loadData(dataDir + "/patterns_train.csv",
        dataDir + "/labels_train.csv",
        dataDir + "/folds_train.csv")

    trainDataSeq.toArray
  }
  /**
   * A dummy model
   */
  val lo = data(0)
  val numd = ChainDemo.featureFn(lo.pattern, lo.label).size
  val model: StructSVMModel[X, Y] =
    new StructSVMModel[X, Y](DenseVector.zeros(numd), 0.0,
      DenseVector.zeros(numd), dissolveFunctions, 1)

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

/**
 * @author torekond
 */
abstract class UnitSpec extends FlatSpec with Matchers with OptionValues with Inside with Inspectors {

  val DissolveAdapter = ImageTestAdapter

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