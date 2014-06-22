/**
 *
 */
package ch.ethz.dal.dbcfw.classification

import breeze.linalg._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.optimization.SSGSolver

/**
 *
 */
class StructSVMWithSSG(
  // val sc: SparkContext,
  // val patterns: Vector[Vector[Double]],
  // val labels: Vector[Vector[Double]],
  val data: Vector[LabeledObject],
  // val data: RDD[LabeledObject], // Consists of tuples (Label, Pattern)
  val featureFn: (Vector[Double], Matrix[Double]) ⇒ Vector[Double], // (y, x) => FeatureVector
  val lossFn: (Vector[Double], Vector[Double]) ⇒ Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) ⇒ Vector[Double], // (model, y_i, x_i) => Label
  val predictFn: (StructSVMModel, Matrix[Double]) ⇒ Vector[Double]) {

  var lambda: Double = 1.0
  var numPasses: Int = 2

  def withRegularizer(lambda: Double): StructSVMWithSSG = {
    this.lambda = lambda
    this
  }

  def withNumPasses(numPasses: Int): StructSVMWithSSG = {
    this.numPasses = numPasses
    this
  }

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
      lambda,
      numPasses).optimize()

}