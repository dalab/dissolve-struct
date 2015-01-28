package ch.ethz.dalab.dissolve.optimization

import breeze.linalg._
import breeze.numerics._
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.regression.LabeledObject
import java.io.File
import java.io.PrintWriter

/**
 * Train a structured SVM using standard Stochastic (Sub)Gradient Descent (SGD).
 * The implementation here is single machine, not distributed.
 * 
 * Input:
 * Each data point (x_i, y_i) is composed of:
 * x_i, the data example
 * y_i, the label
 *
 * @param <X> type for the data examples
 * @param <Y> type for the labels of each example
 */
class SSGSolver[X, Y](
  val data: Seq[LabeledObject[X, Y]],
  val featureFn: (X, Y) => Vector[Double], // (x, y) => FeatureVector
  val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel[X, Y], X, Y) => Y, // (model, x_i, y_i) => Label
  val predictFn: (StructSVMModel[X, Y], X) => Y,
  val solverOptions: SolverOptions[X, Y]) {

  val numPasses = solverOptions.numPasses
  val lambda = solverOptions.lambda
  val debugOn: Boolean = solverOptions.debug

  val maxOracle = oracleFn
  val phi = featureFn
  // Number of dimensions of \phi(x, y)
  val ndims: Int = phi(data(0).pattern, data(0).label).size

  // Filenames
  val lossWriterFileName = "data/debug/ssg-loss.csv"

  /**
   * SSG optimizer
   */
  def optimize(): StructSVMModel[X, Y] = {

    var k: Integer = 0
    val n: Int = data.length
    val d: Int = featureFn(data(0).pattern, data(0).label).size
    // Use first example to determine dimension of w
    val model: StructSVMModel[X, Y] = new StructSVMModel(DenseVector.zeros(featureFn(data(0).pattern, data(0).label).size),
      0.0,
      DenseVector.zeros(ndims),
      featureFn,
      lossFn,
      oracleFn,
      predictFn)

    // Initialization in case of Weighted Averaging
    var wAvg: DenseVector[Double] =
      if (solverOptions.doWeightedAveraging)
        DenseVector.zeros(d)
      else null

    var debugIter = if (solverOptions.debugMultiplier == 0) {
      solverOptions.debugMultiplier = 100
      n
    } else {
      1
    }
    val debugModel: StructSVMModel[X, Y] = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(ndims), featureFn, lossFn, oracleFn, predictFn)

    val lossWriter = if (solverOptions.debugLoss) new PrintWriter(new File(lossWriterFileName)) else null
    if (solverOptions.debugLoss) {
      if (solverOptions.testData != null)
        lossWriter.write("pass_num,iter,primal,dual,duality_gap,train_error,test_error\n")
      else
        lossWriter.write("pass_num,iter,primal,dual,duality_gap,train_error\n")
    }

    if (debugOn) {
      println("Beginning training of %d data points in %d passes with lambda=%f".format(n, numPasses, lambda))
    }

    for (passNum <- 0 until numPasses) {

      if (debugOn)
        println("Starting pass #%d".format(passNum))

      for (dummy <- 0 until n) {
        // 1) Pick example
        val i: Int = dummy
        val pattern: X = data(i).pattern
        val label: Y = data(i).label

        // 2) Solve loss-augmented inference for point i
        val ystar_i: Y = maxOracle(model, pattern, label)

        // 3) Get the subgradient
        val psi_i: Vector[Double] = phi(pattern, label) - phi(pattern, ystar_i)
        val w_s: Vector[Double] = psi_i :* (1 / (n * lambda))

        if (debugOn && dummy == (n - 1))
          csvwrite(new File("data/debug/scala-w-%d.csv".format(passNum + 1)), w_s.toDenseVector.toDenseMatrix)

        // 4) Step size gamma
        val gamma: Double = 1.0 / (k + 1.0)

        // 5) Update the weights of the model
        val newWeights: Vector[Double] = (model.getWeights() :* (1 - gamma)) + (w_s :* (gamma * n))
        model.updateWeights(newWeights)

        k = k + 1

        if (solverOptions.doWeightedAveraging) {
          val rho: Double = 2.0 / (k + 2.0)
          wAvg = wAvg * (1.0 - rho) + model.getWeights() * rho
        }

        /*if (debugOn && k >= debugIter) {
          if (solverOptions.doWeightedAveraging) {
            debugModel.updateWeights(wAvg)
          } else {
            debugModel.updateWeights(model.getWeights)
          }

          val primal = SolverUtils.primalObjective(data, featureFn, lossFn, oracleFn, debugModel, lambda)
          val trainError = SolverUtils.averageLoss(data, lossFn, predictFn, debugModel)

          if (solverOptions.testData != null) {
            val testError = SolverUtils.averageLoss(solverOptions.testData, lossFn, predictFn, debugModel)
            println("Pass %d Iteration %d, SVM primal = %f, Train error = %f, Test error = %f"
              .format(passNum + 1, k, primal, trainError, testError))

            if (solverOptions.debugLoss)
              lossWriter.write("%d,%d,%f,%f,%f\n".format(passNum + 1, k, primal, trainError, testError))
          } else {
            println("Pass %d Iteration %d, SVM primal = %f, Train error = %f"
              .format(passNum + 1, k, primal, trainError))
            if (solverOptions.debugLoss)
              lossWriter.write("%d,%d,%f,%f,\n".format(passNum + 1, k, primal, trainError))
          }

          debugIter = min(debugIter + n, ceil(debugIter * (1 + solverOptions.debugMultiplier / 100)))
        }*/

      }
      if (debugOn)
        println("Completed pass #%d".format(passNum))

    }

    return model
  }

}