/**
 *
 */
package ch.ethz.dalab.dissolve.classification

import breeze.linalg._
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions

/**
 * This is analogous to the model:
 * a) returned by Andrea Vedaldi's svm-struct-matlab
 * b) used by BCFWStruct. See (Lacoste-Julien, Jaggi, Schmidt, Pletscher; ICML 2013)
 *
 * @constructor Create a new StructSVM model
 * @param weights Primal variable. Corresponds to w in Algorithm 4
 * @param ell Corresponds to l in Algorithm 4
 * @param ellMat Corresponds to l_i in Algorithm 4
 * @param pred Prediction function
 */
class StructSVMModel[X, Y](
  var weights: Vector[Double],
  var ell: Double,
  val ellMat: Vector[Double],
  val dissolveFunctions: DissolveFunctions[X, Y],
  val numClasses: Int) extends Serializable {

  def this(
    weights: Vector[Double],
    ell: Double,
    ellMat: Vector[Double],
    dissolveFunctions: DissolveFunctions[X, Y]) =
    this(weights, ell, ellMat, dissolveFunctions, -1)

    
  def getWeights(): Vector[Double] = {
    weights
  }

  def setWeights(newWeights: Vector[Double]) = {
    weights = newWeights
  }

  def getEll(): Double =
    ell

  def updateEll(newEll: Double) =
    ell = newEll

  def predict(pattern: X): Y = {
    dissolveFunctions.predictFn(this, pattern)
  }

  override def clone(): StructSVMModel[X, Y] = {
    new StructSVMModel(this.weights.copy,
      ell,
      this.ellMat.copy,
      dissolveFunctions,
      numClasses)
  }
}