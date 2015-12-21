package ch.ethz.dalab.dissolve.models

import scala.collection.mutable.HashMap

import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions

class MulticlassSVM(numClasses: Int, classFreqMap: HashMap[Int, Double]) extends DissolveFunctions[Vector[Double], Int] {

  /**
   * Feature function
   *
   * Analogous to phi(y) in (2)
   * Returns y_i * x_i
   *
   */
  def featureFn(x: Vector[Double], y: Int): Vector[Double] = {
    assert(y < numClasses,
      "numClasses = %d. Found y_i.label = %d"
        .format(numClasses, y))

    val featureVector = Vector.zeros[Double](x.size * numClasses)
    val numDims = x.size

    // Populate the featureVector in blocks [<class-0 features> <class-1 features> ...].
    val startIdx = y * numDims
    val endIdx = startIdx + numDims

    featureVector(startIdx until endIdx) := x

    featureVector
  }

  /**
   * Loss function
   *
   * Returns 0 if yTruth == yPredict, 1 otherwise
   * Equivalent to max(0, 1 - y w^T x)
   */
  def lossFn(yTruth: Int, yPredict: Int): Double =
    if (yTruth == yPredict)
      0.0
    else
      1.0

  /**
   * Maximization Oracle
   *
   * Want: argmax L(y_i, y) - <w, psi_i(y)>
   * This returns the most violating (Loss-augmented) label.
   */
  override def oracleFn(weights: Vector[Double], xi: Vector[Double], yi: Int): Int = {

    // Obtain a list of scores for each class
    val mostViolatedContraint: (Int, Double) =
      (0 until numClasses).map {
        case cl =>
          (cl, weights dot featureFn(xi, cl))
      }.map {
        case (cl, score) =>
          (cl, score + 1.0)
      }.map { // Loss-augment the scores
        case (cl, score) =>
          if (yi == cl)
            (cl, score - 1.0)
          else
            (cl, score)
      }.maxBy { // Obtain the class with the maximum value
        case (cl, score) => score
      }

    mostViolatedContraint._1
  }

  /**
   * Prediction function
   */
  def predictFn(weights: Vector[Double], xi: Vector[Double]): Int = {

    assert(numClasses > 1)

    val prediction =
      (0 until numClasses).map {
        case cl =>
          (cl, weights dot featureFn(xi, cl))
      }.maxBy { // Obtain the class with the maximum value
        case (cl, score) => score
      }

    prediction._1

  }

}