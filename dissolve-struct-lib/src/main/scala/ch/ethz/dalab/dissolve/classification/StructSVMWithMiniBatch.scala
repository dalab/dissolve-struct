package ch.ethz.dalab.dissolve.classification

import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg._
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import ch.ethz.dalab.dissolve.optimization.DBCFWSolver
import scala.reflect.ClassTag

class StructSVMWithMiniBatch[X, Y](
  val data: RDD[LabeledObject[X, Y]],
  val featureFn: (Y, X) => Vector[Double], // (y, x) => FeatureVector
  val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel[X, Y], Y, X) => Y, // (model, y_i, x_i) => Label
  val predictFn: (StructSVMModel[X, Y], X) => Y,
  val solverOptions: SolverOptions[X, Y]) {

  def trainModel()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] =
    new DBCFWSolver(
      data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions,
      miniBatchEnabled = true).optimize()._1
}