
/**
 *
 */
package ch.ethz.dal.dbcfw.classification

import scala.reflect.ClassTag
import java.io.FileWriter
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.optimization.SolverOptions
import ch.ethz.dal.dbcfw.optimization.DBCFWSolver
import ch.ethz.dal.dbcfw.optimization.SolverUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg.{ Vector, SparseVector, DenseVector }
import ch.ethz.dal.dbcfw.optimization.DBCFWSolverTuned

/**
 * @author tribhu
 *
 */
object BinarySVMWithDBCFW {

  /**
   * Feature function
   *
   * Analogous to phi(y) in (2)
   * Returns y_i * x_i
   *
   */
  def featureFn(y: Double, x: Vector[Double]): Vector[Double] = {
    x * y
  }

  /**
   * Loss function
   *
   * Returns 0 if yTruth == yPredict, 1 otherwise
   * Equivalent to max(0, 1 - y w^T x)
   */
  def lossFn(yTruth: Double, yPredict: Double): Double =
    if (yTruth == yPredict)
      0.0
    else
      1.0

  /**
   * Maximization Oracle
   *
   * Want: max L(y_i, y) - <w, psi_i(y)>
   * This returns the most violating (Loss-augmented) label.
   */
  def oracleFn(model: StructSVMModel[Vector[Double], Double], yi: Double, xi: Vector[Double]): Double = {

    val weights = model.getWeights()

    var score_neg1 = weights dot featureFn(-1.0, xi)
    var score_pos1 = weights dot featureFn(1.0, xi)

    // Loss augment the scores
    score_neg1 += 1.0
    score_pos1 += 1.0

    if (yi == -1.0)
      score_neg1 -= 1.0
    else if (yi == 1.0)
      score_pos1 -= 1.0
    else
      throw new IllegalArgumentException("yi not in [-1, 1], yi = " + yi)

    if (score_neg1 > score_pos1)
      -1.0
    else
      1.0
  }

  /**
   * Prediction function
   */
  def predictFn(model: StructSVMModel[Vector[Double], Double], xi: Vector[Double]): Double = {

    val weights = model.getWeights()

    val score_neg1 = weights dot featureFn(-1.0, xi)
    val score_pos1 = weights dot featureFn(1.0, xi)

    if (score_neg1 > score_pos1)
      -1.0
    else
      +1.0

  }

  /**
   * Classifying with in-built functions
   */
  def train(
    data: RDD[LabeledPoint],
    solverOptions: SolverOptions[Vector[Double], Double]): StructSVMModel[Vector[Double], Double] = {

    // Convert the RDD[LabeledPoint] to RDD[LabeledObject]
    val objectifiedData: RDD[LabeledObject[Vector[Double], Double]] =
      data.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], Double](x.label, SparseVector(x.features.toArray))
      }

    val repartData =
      if (solverOptions.enableManualPartitionSize)
        objectifiedData.repartition(solverOptions.NUM_PART)
      else
        objectifiedData

    println(solverOptions)

    val (trainedModel, debugInfo) = new DBCFWSolverTuned[Vector[Double], Double](
      repartData,
      this.featureFn,
      this.lossFn,
      this.oracleFn,
      this.predictFn,
      solverOptions,
      miniBatchEnabled = false).optimize()

    println(debugInfo)

    // Dump debug information into a file
    val fw = new FileWriter(solverOptions.debugInfoPath)
    // Write the current parameters being used
    fw.write(solverOptions.toString())
    fw.write("\n")

    // Write spark-specific parameters
    fw.write(SolverUtils.getSparkConfString(data.context.getConf))
    fw.write("\n")

    // Write values noted from the run
    fw.write(debugInfo)
    fw.close()

    trainedModel

  }

  /**
   * Classifying with user-submitted functions
   */
  def train(
    data: RDD[LabeledPoint],
    featureFn: (Double, Vector[Double]) => Vector[Double], // (y, x) => FeatureVector
    lossFn: (Double, Double) => Double, // (yTruth, yPredict) => LossValue
    oracleFn: (StructSVMModel[Vector[Double], Double], Double, Vector[Double]) => Double, // (model, y_i, x_i) => Label
    predictFn: (StructSVMModel[Vector[Double], Double], Vector[Double]) => Double,
    solverOptions: SolverOptions[Vector[Double], Double]): StructSVMModel[Vector[Double], Double] = {

    // Convert the RDD[LabeledPoint] to RDD[LabeledObject]
    val objectifiedData: RDD[LabeledObject[Vector[Double], Double]] =
      data.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], Double](x.label,
            if (solverOptions.sparse)
              SparseVector(x.features.toArray)
            else
              DenseVector(x.features.toArray))
      }

    val repartData =
      if (solverOptions.enableManualPartitionSize)
        objectifiedData.repartition(solverOptions.NUM_PART)
      else
        objectifiedData

    println(solverOptions)

    val (trainedModel, debugInfo) = new DBCFWSolverTuned[Vector[Double], Double](
      repartData,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions,
      miniBatchEnabled = false).optimize()

    // Dump debug information into a file
    val fw = new FileWriter(solverOptions.debugInfoPath)
    // Write the current parameters being used
    fw.write(solverOptions.toString())
    fw.write("\n")

    // Write spark-specific parameters
    fw.write(SolverUtils.getSparkConfString(data.context.getConf))
    fw.write("\n")

    // Write values noted from the run
    fw.write(debugInfo)
    fw.close()

    println(debugInfo)

    trainedModel

  }

}