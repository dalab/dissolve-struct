package ch.ethz.dal.dbcfw.classification

import ch.ethz.dal.dbcfw.regression.LabeledObject
import breeze.linalg._
import ch.ethz.dal.dbcfw.optimization.SolverOptions
import org.apache.spark.SparkContext
import ch.ethz.dal.dbcfw.optimization.DBCFWSolver

class StructSVMWithDBCFW(
  val sc: SparkContext,
  val data: Vector[LabeledObject],
  val featureFn: (Vector[Double], Matrix[Double]) => Vector[Double], // (y, x) => FeatureVector
  val lossFn: (Vector[Double], Vector[Double]) => Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double], // (model, y_i, x_i) => Label
  val predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
  val solverOptions: SolverOptions) {

  def trainModel(): StructSVMModel =
    new DBCFWSolver(sc,
      data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions,
      miniBatchEnabled=false).optimize()
}