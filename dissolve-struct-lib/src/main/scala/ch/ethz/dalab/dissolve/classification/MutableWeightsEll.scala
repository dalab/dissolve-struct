package ch.ethz.dalab.dissolve.classification

import breeze.linalg._

/**
 * @param weights Primal variable. Corresponds to w in Algorithm 4
 * @param ell Corresponds to l in Algorithm 4
 */
class MutableWeightsEll(
    var weights: Vector[Double],
    var ell: Double) extends Serializable {

  def getWeights(): Vector[Double] = {
    weights
  }

  def setWeights(newWeights: Vector[Double]) = {
    weights = newWeights
  }

  def getEll(): Double =
    ell

  def setEll(newEll: Double) =
    ell = newEll

  override def clone(): MutableWeightsEll = {
    new MutableWeightsEll(this.weights.copy, ell)
  }

  // FIXME Can we make do without this?
  @deprecated
  def +(that: MutableWeightsEll): MutableWeightsEll = {
    new MutableWeightsEll(this.weights + that.weights,
      this.ell + that.ell)
  }

  def +=(that: MutableWeightsEll): MutableWeightsEll = {
    this.weights += that.weights
    this.ell += that.ell

    this
  }
}