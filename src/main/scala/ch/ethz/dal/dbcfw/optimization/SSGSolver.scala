package ch.ethz.dal.dbcfw.optimization

import breeze.linalg._
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import ch.ethz.dal.dbcfw.regression.LabeledObject

/**
 * Input:
 * Each data point (x_i, y_i) is composed of:
 * x_i, the feature Matrix containing Doubles
 * y_i, the label Vector containing Doubles
 *
 * allPatterns is a vector of x_i Matrices
 * allLabels is a vector of y_i Vectors
 */
class SSGSolver(
  // val allPatterns: Vector[Matrix[Double]],
  // val allLabels: Vector[Vector[Double]],
  val data: Vector[LabeledObject],
  val featureFn: (Vector[Double], Matrix[Double]) ⇒ Vector[Double], // (y, x) => FeatureVector
  val lossFn: (Vector[Double], Vector[Double]) ⇒ Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) ⇒ Vector[Double], // (model, y_i, x_i) => Label
  val predictFn: (StructSVMModel, Matrix[Double]) ⇒ Vector[Double],
  // Parameters
  val lambda: Double,
  // val gapThreshold: Double,
  // val gapCheck: Boolean,
  val numPasses: Integer /*,
  val doLinesearch: Boolean,
  val doWeightedAveraging: Boolean,
  val timeBudget: Integer,
  val randSeed: Integer,
  val sample: String*/ ) {

  val maxOracle = oracleFn
  val phi = featureFn
  val debugOn: Boolean = true;
  // Number of dimensions of \phi(x, y)
  val ndims: Int = phi(data(0).label, data(0).pattern).size

  /**
   * SSG optimizer
   */
  def optimize(): StructSVMModel = {

    var k: Integer = 0
    val n: Int = data.length
    val model: StructSVMModel = new StructSVMModel(DenseVector.zeros(ndims),
      0.0,
      DenseVector.zeros(ndims),
      featureFn,
      lossFn,
      oracleFn,
      predictFn)

    if (debugOn) {
      println("Beginning training of %d data points in %d passes with lambda=%f".format(n, numPasses, lambda))
    }

    for (passNum ← 0 until numPasses) {

      if (debugOn)
        println("Starting pass #%d".format(passNum))

      for (dummy ← 1 until n) {
        // 1) Pick example
        val i: Int = dummy
        val pattern: Matrix[Double] = data(i).pattern
        val label: Vector[Double] = data(i).label

        // 2) Solve loss-augmented inference for point i
        val ystar_i: Vector[Double] = maxOracle(model, label, pattern)

        // 3) Get the subgradient
        val psi_i: Vector[Double] = phi(label, pattern) - phi(ystar_i, pattern)
        val w_s: Vector[Double] = psi_i :* (1 / n * lambda);

        // 4) Step size gamma
        val gamma: Double = 1 / (k + 1)

        // 5) Update the weights of the model
        val newWeights: Vector[Double] = (model.getWeights() :* (1 - gamma)) :+ (w_s :* gamma)
        model.updateWeights(newWeights)

        k = k + 1

      }
      if (debugOn)
        println("Completed pass #%d".format(passNum))

    }

    return model
  }

}