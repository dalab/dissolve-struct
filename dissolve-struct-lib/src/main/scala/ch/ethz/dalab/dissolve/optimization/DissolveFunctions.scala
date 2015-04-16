package ch.ethz.dalab.dissolve.optimization

import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.classification.StructSVMModel

trait DissolveFunctions[X, Y] extends Serializable {

  def featureFn(x: X, y: Y): Vector[Double]

  def lossFn(yPredicted: Y, yTruth: Y): Double

  def oracleFn(model: StructSVMModel[X, Y], x: X, y: Y): Y =
    oracleCandidateStream(model, x, y).head

  def predictFn(model: StructSVMModel[X, Y], x: X): Y

  def oracleCandidateStream(model: StructSVMModel[X, Y], x: X, y: Y): Stream[Y] =
    oracleFn(model, x, y) #:: Stream.empty

}