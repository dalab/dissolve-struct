/**
 *
 */
package ch.ethz.dal.dbcfw.optimization

import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.rdd.RDD
import scala.util.Random
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import breeze.linalg._
import breeze.numerics._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import java.io.File
import java.io.PrintWriter
import scala.collection.mutable.MutableList

/**
 *
 */
class BCFWSolver /*extends Optimizer*/ (
  val data: Vector[LabeledObject],
  val featureFn: (Vector[Double], Matrix[Double]) => Vector[Double], // (y, x) => FeatureVect, 
  val lossFn: (Vector[Double], Vector[Double]) => Double, // (yTruth, yPredict) => LossVal, 
  val oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double], // (model, y_i, x_i) => Lab, 
  val predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
  val solverOptions: SolverOptions,
  val testData: Vector[LabeledObject]) {

  // Constructor without test data
  def this(
    data: Vector[LabeledObject],
    featureFn: (Vector[Double], Matrix[Double]) => Vector[Double], // (y, x) => FeatureVect, 
    lossFn: (Vector[Double], Vector[Double]) => Double, // (yTruth, yPredict) => LossVal, 
    oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double], // (model, y_i, x_i) => Lab, 
    predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
    solverOptions: SolverOptions) = this(data, featureFn, lossFn, oracleFn, predictFn, solverOptions, null)

  val numPasses = solverOptions.numPasses
  val lambda = solverOptions.lambda
  val debugOn: Boolean = solverOptions.debug
  val xldebug: Boolean = solverOptions.xldebug

  val maxOracle = oracleFn
  val phi = featureFn
  // Number of dimensions of \phi(x, y)
  val ndims: Int = phi(data(0).label, data(0).pattern).size

  val eps: Double = 2.2204E-16
  val lossWriterFileName = "data/debug/bcfw-loss.csv"

  /**
   * BCFW optimizer
   */
  def optimize(): StructSVMModel = {

    /* Initialization */
    var k: Integer = 0
    val n: Int = data.length
    val d: Int = featureFn(data(0).label, data(0).pattern).size
    // Use first example to determine dimension of w
    val model: StructSVMModel = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)
    val wMat: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, n)
    var ell: Double = 0.0
    val ellMat: DenseVector[Double] = DenseVector.zeros[Double](n)

    // Initialization in case of Weighted Averaging
    var wAvg: DenseVector[Double] =
      if (solverOptions.doWeightedAveraging)
        DenseVector.zeros(d)
      else null
    var lAvg: Double = 0.0

    var debugIter = if (solverOptions.debugMultiplier == 0) {
      solverOptions.debugMultiplier = 100
      n
    } else {
      1
    }
    val debugModel: StructSVMModel = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(ndims), featureFn, lossFn, oracleFn, predictFn)

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

    // Initialize the cache: Index -> List of precomputed ystar_i's
    var oracleCache = collection.mutable.Map[Int, MutableList[Vector[Double]]]()

    for (passNum <- 0 until numPasses) {

      if (debugOn)
        println("Starting pass #%d".format(passNum))

      for (dummy <- 0 until n) {
        // 1) Pick example
        val i: Int = dummy
        val pattern: Matrix[Double] = data(i).pattern
        val label: Vector[Double] = data(i).label

        // 2) Solve loss-augmented inference for point i
        // 2.a) If cache is enabled, check if any of the previous ystar_i's for this i can be used
        val cachedYstar_i: Option[Vector[Double]] =
          if (solverOptions.enableOracleCache && oracleCache.contains(i)) {
            val contenders: Seq[(Double, Int)] =
              oracleCache(i)
                .map(y_i => (((phi(label, pattern) - phi(y_i, pattern)) :* (1 / (n * lambda))),
                  (1.0 / n) * lossFn(label, y_i))) // Map each cached y_i to their respective (w_s, ell_s)
                .map {
                  case (w_s, ell_s) => (model.getWeights().t * (wMat(::, i) - w_s) - ((ellMat(i) - ell_s) * (1 / lambda))) /
                    ((wMat(::, i) - w_s).t * (wMat(::, i) - w_s) + eps) // Map each (w_s, ell_s) to their respective step-size values	
                }
                .zipWithIndex // We'll need the index later to retrieve the respective approx. ystar_i
                .filter { case (gamma, idx) => gamma > 0.0 }
                .map { case (gamma, idx) => (min(1.0, gamma), idx) } // Clip to [0,1] interval
                .sortBy { case (gamma, idx) => gamma }

            // TODO Use this naive_gamma to further narrow down on cached contenders
            // TODO Maintain fixed size of the list of cached vectors
            val naive_gamma: Double = (2.0 * n) / (k + 2.0 * n)

            // If there is a good contender among the cached datapoints, return it
            if (contenders.size >= 1)
              Some(oracleCache(i)(contenders.head._2))
            else None
          } else
            None

        // 2.b) In case cache is disabled or a good contender from cache hasn't been found, call max Oracle
        val ystar_i: Vector[Double] =
          if (cachedYstar_i.isEmpty) {
            val ystar = maxOracle(model, label, pattern)

            if (solverOptions.enableOracleCache)
              // Add this newly computed ystar to the cache of this i
              oracleCache.update(i, oracleCache.getOrElse(i, MutableList[Vector[Double]]()) :+ ystar)

            ystar
          } else {
            cachedYstar_i.get
          }

        // 3) Define the update quantities
        val psi_i: Vector[Double] = phi(label, pattern) - phi(ystar_i, pattern)
        val w_s: Vector[Double] = psi_i :* (1 / (n * lambda))
        val loss_i: Double = lossFn(label, ystar_i)
        val ell_s: Double = (1.0 / n) * loss_i

        // 4) Get step-size gamma
        val gamma: Double =
          if (solverOptions.doLineSearch) {
            val gamma_opt = (model.getWeights().t * (wMat(::, i) - w_s) - ((ellMat(i) - ell_s) * (1 / lambda))) /
              ((wMat(::, i) - w_s).t * (wMat(::, i) - w_s) + eps)
            max(0.0, min(1.0, gamma_opt))
          } else {
            (2.0 * n) / (k + 2.0 * n)
          }

        // 5, 6, 7, 8) Update the weights of the model
        val tempWeights1: Vector[Double] = model.getWeights() - wMat(::, i)
        model.updateWeights(tempWeights1)
        wMat(::, i) := (wMat(::, i) * (1.0 - gamma)) + (w_s * gamma)
        val tempWeights2: Vector[Double] = model.getWeights() + wMat(::, i)
        model.updateWeights(tempWeights2)

        ell = ell - ellMat(i)
        ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma)
        ell = ell + ellMat(i)

        // 9) Optionally update the weighted average
        if (solverOptions.doWeightedAveraging) {
          val rho: Double = 2.0 / (k + 2.0)
          wAvg = (wAvg * (1.0 - rho)) + (model.getWeights * rho)
          lAvg = (lAvg * (1.0 - rho)) + (ell * rho)
        }

        /**
         * DEBUG/TEST code
         */
        // If this is the last pass and debugWeights flag is true, dump weight vector to CSV
        if (solverOptions.debugWeights && dummy == (n - 1))
          csvwrite(new File("data/debug/debugWeights/scala-w-%d.csv".format(passNum + 1)),
            { if (solverOptions.doWeightedAveraging) wAvg else model.getWeights }.toDenseVector.toDenseMatrix)

        k = k + 1

        if (debugOn && k >= debugIter) {
          if (solverOptions.doWeightedAveraging) {
            debugModel.updateWeights(wAvg)
            debugModel.updateEll(lAvg)
          } else {
            debugModel.updateWeights(model.getWeights)
            debugModel.updateEll(model.getEll)
          }
          val f = -SolverUtils.objectiveFunction(debugModel.getWeights, debugModel.getEll, lambda)
          val gapTup = SolverUtils.dualityGap(data, phi, lossFn, maxOracle, debugModel, lambda)
          val gap = gapTup._1
          val primal = f + gap
          val trainError = SolverUtils.averageLoss(data, lossFn, predictFn, debugModel)

          if (solverOptions.testData != null) {
            val testError = SolverUtils.averageLoss(solverOptions.testData, lossFn, predictFn, debugModel)
            println("Pass %d Iteration %d, SVM primal = %f, SVM dual = %f, Duality gap = %f, Train error = %f, Test error = %f"
              .format(passNum + 1, k, primal, f, gap, trainError, testError))

            if (solverOptions.debugLoss)
              lossWriter.write("%d,%d,%f,%f,%f,%f,%f\n".format(passNum + 1, k, primal, f, gap, trainError, testError))
          } else {
            println("Pass %d Iteration %d, SVM primal = %f, SVM dual = %f, Duality gap = %f, Train error = %f"
              .format(passNum + 1, k, primal, f, gap, trainError))
            if (solverOptions.debugLoss)
              lossWriter.write("%d,%d,%f,%f,%f,%f\n".format(passNum + 1, k, primal, f, gap, trainError))
          }

          debugIter = min(debugIter + n, ceil(debugIter * (1 + solverOptions.debugMultiplier / 100)))
        }

      }
      if (debugOn)
        println("Completed pass #%d".format(passNum))

    }

    if (solverOptions.doWeightedAveraging) {
      model.updateWeights(wAvg)
      model.updateEll(lAvg)
    } else {
      model.updateEll(ell)
    }

    if (lossWriter != null)
      lossWriter.close()

    return model
  }

}