/**
 *
 */
package ch.ethz.dal.dbcfw.classification

import breeze.linalg._

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
class StructSVMModel(
  var weights: Vector[Double],
  var ell: Double,
  val ellMat: Vector[Double],
  val featureFn: (Vector[Double], Matrix[Double]) => Vector[Double],
  val lossFn: (Vector[Double], Vector[Double]) => Double,
  val oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double],
  val predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double]) extends Serializable {

  def getWeights(): Vector[Double] = {
    weights
  }

  def updateWeights(newWeights: Vector[Double]) = {
    weights = newWeights
  }

  def getEll(): Double =
    ell

  def updateEll(newEll: Double) =
    ell = newEll

  override def clone(): StructSVMModel = {
    new StructSVMModel(this.weights.copy,
      ell,
      this.ellMat.copy,
      featureFn,
      lossFn,
      oracleFn,
      predictFn)
  }
}