package ch.ethz.dalab.dissolve.classification

import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg._
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import ch.ethz.dalab.dissolve.optimization.DBCFWSolver
import java.io.FileWriter
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import scala.reflect.ClassTag
import ch.ethz.dalab.dissolve.optimization.DBCFWSolverTuned

class StructSVMWithDBCFW[X, Y](
  val data: RDD[LabeledObject[X, Y]],
  val featureFn: (X, Y) => Vector[Double], // (x, y) => FeatureVector
  val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel[X, Y], X, Y) => Y, // (model, x_i, y_i) => Label
  val predictFn: (StructSVMModel[X, Y], X) => Y,
  val solverOptions: SolverOptions[X, Y]) {

  def trainModel()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {
    val (trainedModel, debugInfo) = new DBCFWSolverTuned[X, Y](
      data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions,
      miniBatchEnabled = false).optimize()

    // Dump debug information into a file
    val fw = new FileWriter(solverOptions.debugInfoPath)
    // Write the current parameters being used
    fw.write(solverOptions.toString())
    fw.write("\n")

    // Write spark-specific parameters
    fw.write(SolverUtils.getSparkConfString(data.context.getConf))
    fw.write("\n")

    // Write values noted from the run
    fw.write(debugInfo)
    fw.close()

    print(debugInfo)

    // Return the trained model
    trainedModel
  }
}

object StructSVMWithDBCFW {
  def train[X, Y](data: RDD[LabeledObject[X, Y]],
                  featureFn: (X, Y) => Vector[Double], // (x, y) => FeatureVector
                  lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossValue
                  oracleFn: (StructSVMModel[X, Y], X, Y) => Y, // (model, x_i, y_i) => Label
                  predictFn: (StructSVMModel[X, Y], X) => Y,
                  solverOptions: SolverOptions[X, Y])(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {
    val (trainedModel, debugInfo) = new DBCFWSolverTuned[X, Y](
      data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions,
      miniBatchEnabled = false).optimize()

    // Dump debug information into a file
    val fw = new FileWriter(solverOptions.debugInfoPath)
    // Write the current parameters being used
    fw.write(solverOptions.toString())
    fw.write("\n")

    // Write spark-specific parameters
    fw.write(SolverUtils.getSparkConfString(data.context.getConf))
    fw.write("\n")

    // Write values noted from the run
    fw.write(debugInfo)
    fw.close()

    print(debugInfo)

    // Return the trained model
    trainedModel

  }
}