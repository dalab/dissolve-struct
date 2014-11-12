/**
 *
 */
package ch.ethz.dal.dbcfw.classification

import breeze.linalg._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.optimization.SSGSolver
import ch.ethz.dal.dbcfw.optimization.SolverOptions
import scala.reflect.ClassTag

/**
 *
 */
class StructSVMWithSSG[X, Y](
  // val sc: SparkContext,
  // val patterns: Vector[Vector[Double]],
  // val labels: Vector[Vector[Double]],
  val data: Vector[LabeledObject[X, Y]],
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
    new SSGSolver(data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions).optimize()

}