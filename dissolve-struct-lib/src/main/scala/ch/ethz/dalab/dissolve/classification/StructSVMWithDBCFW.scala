package ch.ethz.dalab.dissolve.classification

import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg._
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import java.io.FileWriter
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import scala.reflect.ClassTag
import ch.ethz.dalab.dissolve.optimization.DBCFWSolverTuned
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions

class StructSVMWithDBCFW[X, Y](
  val data: RDD[LabeledObject[X, Y]],
  val dissolveFunctions: DissolveFunctions[X, Y],
  val solverOptions: SolverOptions[X, Y]) {

  def trainModel()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {
    val (trainedModel, debugInfo) = new DBCFWSolverTuned[X, Y](
      data,
      dissolveFunctions,
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
                  dissolveFunctions: DissolveFunctions[X, Y],
                  solverOptions: SolverOptions[X, Y])(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {
    val (trainedModel, debugInfo) = new DBCFWSolverTuned[X, Y](
      data,
      dissolveFunctions,
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