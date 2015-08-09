package ch.ethz.dalab.dissolve.optimization

import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.classification.StructSVMModel

trait DissolveFunctions[X, Y] extends Serializable {

  def featureFn(x: X, y: Y): Vector[Double]

  def lossFn(yPredicted: Y, yTruth: Y): Double

  // Override either `oracleFn` or `oracleCandidateStream`
  def oracleFn(model: StructSVMModel[X, Y], x: X, y: Y): Y =
    oracleCandidateStream(model, x, y).head

  def oracleCandidateStream(model: StructSVMModel[X, Y], x: X, y: Y): Stream[Y] =
    oracleFn(model, x, y) #:: Stream.empty

  def predictFn(model: StructSVMModel[X, Y], x: X): Y

  /**
   * Image Segmentation-specific adapters
   */
  def getImageID(x: X): String = "NA"
  
  def numClasses(): Int = -1

  // Returns: Label -> (TotalPixelCount, CorrectPixelCount)
  def perClassAccuracy(yPredicted: Y, yTruth: Y): Array[(Int, Int)] = null

}