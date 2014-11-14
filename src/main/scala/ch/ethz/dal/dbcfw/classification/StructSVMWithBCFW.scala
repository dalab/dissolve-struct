/**
 *
 */
package ch.ethz.dal.dbcfw.classification

import org.apache.spark.SparkContext
import breeze.linalg._
import org.apache.spark.rdd.RDD
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.optimization.SolverOptions
import ch.ethz.dal.dbcfw.optimization.BCFWSolver
import scala.reflect.ClassTag

/**
 * Analogous to BCFWSolver
 *
 * TODO
 * - Replace (patterns, labels) with RDD[StructLabeledPoint]
 *
 *
 */
class StructSVMWithBCFW[X, Y](
  val data: Seq[LabeledObject[X, Y]],
  val featureFn: (Y, X) => Vector[Double], // (y, x) => FeatureVector
  val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel[X, Y], Y, X) => Y, // (model, y_i, x_i) => Label
  val predictFn: (StructSVMModel[X, Y], X) => Y,
  val solverOptions: SolverOptions[X, Y]) {

  /*val optimizer: SSGSolver = new SSGSolver(data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      lambda,
      numPasses)*/

  def trainModel()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] =
    new BCFWSolver(data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions).optimize()

}