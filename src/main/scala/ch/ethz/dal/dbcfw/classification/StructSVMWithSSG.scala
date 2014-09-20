/**
 *
 */
package ch.ethz.dal.dbcfw.classification

import breeze.linalg._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.optimization.SSGSolver
import ch.ethz.dal.dbcfw.optimization.SolverOptions

/**
 *
 */
class StructSVMWithSSG(
  // val sc: SparkContext,
  // val patterns: Vector[Vector[Double]],
  // val labels: Vector[Vector[Double]],
  val data: Vector[LabeledObject],
  // val data: RDD[LabeledObject], // Consists of tuples (Label, Pattern)
  val featureFn: (Vector[Double], Matrix[Double]) => Vector[Double], // (y, x) => FeatureVector
  val lossFn: (Vector[Double], Vector[Double]) => Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double], // (model, y_i, x_i) => Label
  val predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
  val solverOptions: SolverOptions) {

  /*val optimizer: SSGSolver = new SSGSolver(data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      lambda,
      numPasses)*/

  def trainModel(): StructSVMModel =
    new SSGSolver(data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions).optimize()

}