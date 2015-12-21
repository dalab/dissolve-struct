package ch.ethz.dalab.dissolve.models

import scala.collection.mutable.HashMap

import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions

class BinarySVM(classFreqMap: HashMap[Int, Double]) extends DissolveFunctions[Vector[Double], Int] {

  def getClassFreqMap() = classFreqMap

  def setClassFreq(label: Int, freq: Double) = {
    assert(label == 1 || label == -1)
    classFreqMap(label) = freq
  }

  /**
   * Feature function
   *
   * Analogous to phi(y) in (2)
   * Returns y_i * x_i
   *
   */
  def featureFn(x: Vector[Double], y: Int): Vector[Double] = {
    x * y.toDouble
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
   * Want: max L(y_i, y) - <w, psi_i(y)>
   * This returns the most violating (Loss-augmented) label.
   */
  override def oracleFn(weights: Vector[Double], xi: Vector[Double], yi: Int): Int = {

    var score_neg1 = weights dot featureFn(xi, -1)
    var score_pos1 = weights dot featureFn(xi, 1)

    // Loss augment the scores
    score_neg1 += 1.0
    score_pos1 += 1.0

    if (yi == -1)
      score_neg1 -= 1.0
    else if (yi == 1)
      score_pos1 -= 1.0
    else
      throw new IllegalArgumentException("yi not in [-1, 1], yi = " + yi)

    if (score_neg1 > score_pos1)
      -1
    else
      1
  }

  /**
   * Prediction function
   */
  def predictFn(weights: Vector[Double], xi: Vector[Double]): Int = {

    val score_neg1 = weights dot featureFn(xi, -1)
    val score_pos1 = weights dot featureFn(xi, 1)

    if (score_neg1 > score_pos1)
      -1
    else
      +1

  }

}