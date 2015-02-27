/**
 *
 */
package ch.ethz.dalab.dissolve.classification

import scala.reflect.ClassTag

import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.optimization.SSGSolver
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.regression.LabeledObject

/**
 *
 */
class StructSVMWithSSG[X, Y](
  val data: Seq[LabeledObject[X, Y]],
  val dissolveFunctions: DissolveFunctions[X, Y],
  val solverOptions: SolverOptions[X, Y]) {

  def trainModel()(implicit m: ClassTag[Y]): StructSVMModel[X, Y] =
    new SSGSolver(data,
      dissolveFunctions,
      solverOptions).optimize()

}