/**
 *
 */
package ch.ethz.dalab.dissolve.classification

import java.io.FileWriter

import scala.reflect.ClassTag

import ch.ethz.dalab.dissolve.optimization.BCFWSolver
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.regression.LabeledObject

/**
 * Analogous to BCFWSolver
 *
 *
 */
class StructSVMWithBCFW[X, Y](
  val data: Seq[LabeledObject[X, Y]],
  val dissolveFunctions: DissolveFunctions[X, Y],
  val solverOptions: SolverOptions[X, Y]) {

  def trainModel()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {
    val (trainedModel, debugInfo) = new BCFWSolver(data,
      dissolveFunctions,
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