/**
 *
 */
package ch.ethz.dal.dbcfw.classification

import scala.reflect.ClassTag

import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.optimization.SolverOptions
import ch.ethz.dal.dbcfw.optimization.DBCFWSolver

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

import breeze.linalg.{ Vector, SparseVector }

class BinarySVMWithDBCFW[X, Y] private (val data: RDD[LabeledObject[X, Y]],
                                        val featureFn: (Y, X) => Vector[Double], // (y, x) => FeatureVector
                                        val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossValue
                                        val oracleFn: (StructSVMModel[X, Y], Y, X) => Y, // (model, y_i, x_i) => Label
                                        val predictFn: (StructSVMModel[X, Y], X) => Y,
                                        val solverOptions: SolverOptions[X, Y]) {

  def run()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {
    val (trainedModel, debugInfo) = new DBCFWSolver[X, Y](
      data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions,
      miniBatchEnabled = false).optimize()

    trainedModel
  }

}

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
   * This returns a negative value in case of a correct prediction and positive in case of a false prediction
   */
  def oracleFn(model: StructSVMModel[Vector[Double], Double], yi: Double, xi: Vector[Double]): Double = {

    val yPredict = xi dot model.getWeights()
    lossFn(yi, yPredict)
  }

  /**
   * Prediction function
   */
  def predictFn(model: StructSVMModel[Vector[Double], Double], xi: Vector[Double]): Double = {

    val weights = model.getWeights()

    if ((weights dot featureFn(1.0, xi)) > (weights dot featureFn(-1.0, xi)))
      1.0
    else
      -1.0
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
          new LabeledObject[Vector[Double], Double](x.label, SparseVector(x.features.toArray)) // Is the asInstanceOf required?
      }

    val (trainedModel, debugInfo) = new DBCFWSolver[Vector[Double], Double](
      objectifiedData,
      this.featureFn,
      this.lossFn,
      this.oracleFn,
      this.predictFn,
      solverOptions,
      miniBatchEnabled = false).optimize()

    println(debugInfo)

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
          new LabeledObject[Vector[Double], Double](x.label, SparseVector(x.features.toArray))
      }

    val (trainedModel, debugInfo) = new DBCFWSolver[Vector[Double], Double](
      objectifiedData,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions,
      miniBatchEnabled = false).optimize()

    println(debugInfo)

    trainedModel

  }

}