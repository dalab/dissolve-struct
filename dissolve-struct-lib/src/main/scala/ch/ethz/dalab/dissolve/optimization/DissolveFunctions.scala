package ch.ethz.dalab.dissolve.optimization

import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.classification.MutableWeightsEll

trait DissolveFunctions[X, Y] extends Serializable {

  def featureFn(x: X, y: Y): Vector[Double]

  def lossFn(yPredicted: Y, yTruth: Y): Double

  // Override either `oracleFn` or `oracleCandidateStream`
  def oracleFn(weights: Vector[Double], x: X, y: Y): Y =
    oracleCandidateStream(weights, x, y).head

  def oracleCandidateStream(weights: Vector[Double], x: X, y: Y): Stream[Y] =
    oracleFn(weights, x, y) #:: Stream.empty

  def predictFn(weights: Vector[Double], x: X): Y

  def classWeights(y: Y): Double = 1.0

}