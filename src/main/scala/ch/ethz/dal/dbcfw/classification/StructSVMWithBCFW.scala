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

/**
 * Analogous to BCFWSolver
 *
 * TODO
 * - Replace (patterns, labels) with RDD[StructLabeledPoint]
 *
 *
 */
class StructSVMWithBCFW(
  val data: Vector[LabeledObject],
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
    new BCFWSolver(data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions).optimize()

}