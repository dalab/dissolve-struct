package ch.ethz.dalab.dissolve.classification

import breeze.linalg._
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.optimization.PerceptronSolver
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import scala.reflect.ClassTag


class StructPerceptron[X, Y](
  val data: Seq[LabeledObject[X, Y]],
  val featureFn: (X, Y) => Vector[Double], // (x, y) => FeatureVector
  val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel[X, Y], X, Y) => Y, // (model, x_i, y_i) => Label
  val predictFn: (StructSVMModel[X, Y], X) => Y,
  val solverOptions: SolverOptions[X, Y]) {
  
  def trainModel()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {
    val trainedModel = new PerceptronSolver(data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions).optimize()
      trainedModel
  }
}
