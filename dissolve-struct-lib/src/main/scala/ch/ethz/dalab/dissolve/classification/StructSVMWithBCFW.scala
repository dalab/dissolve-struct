/**
 *
 */
package ch.ethz.dalab.dissolve.classification

import org.apache.spark.SparkContext
import breeze.linalg._
import org.apache.spark.rdd.RDD
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.optimization.BCFWSolver
import scala.reflect.ClassTag
import java.io.FileWriter

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

  def trainModel()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {
    val (trainedModel, debugInfo) = new BCFWSolver(data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions).optimize()

    // Dump debug information into a file
    val fw = new FileWriter(solverOptions.debugInfoPath)
    // Write the current parameters being used
    fw.write("# BCFW\n")
    fw.write(solverOptions.toString())
    fw.write("\n")

    // Write values noted from the run
    fw.write(debugInfo)
    fw.close()

    trainedModel
  }

}