package ch.ethz.dalab.dissolve.classification

import scala.reflect.ClassTag

import org.apache.spark.rdd.RDD

import ch.ethz.dalab.dissolve.optimization.DBCFWSolverTuned
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.regression.LabeledObject

class StructSVMWithMiniBatch[X, Y](
  val data: RDD[LabeledObject[X, Y]],
  val dissolveFunctions: DissolveFunctions[X, Y],
  val solverOptions: SolverOptions[X, Y]) {

  def trainModel()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] =
    new DBCFWSolverTuned(
      data,
      dissolveFunctions,
      solverOptions,
      miniBatchEnabled = true).optimize()._1
}