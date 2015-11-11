/**
 *
 */
package ch.ethz.dalab.dissolve.optimization

import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.rdd.RDD

import scala.util.Random
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import breeze.linalg._
import breeze.numerics._
import ch.ethz.dalab.dissolve.optimization.SolverUtils;
import ch.ethz.dalab.dissolve.regression.LabeledObject

import java.io.File
import java.io.PrintWriter

import scala.collection.mutable.MutableList
import scala.reflect.ClassTag

/**
 * Train a structured SVM using the primal dual Block-Coordinate Frank-Wolfe solver (BCFW).
 *
 * The implementation here is single machine, not distributed. See DBCFWSolver... for the
 * distributed version.
 *
 * @param <X> type for the data examples
 * @param <Y> type for the labels of each example
 */
class BCFWSolver[X, Y] /*extends Optimizer*/ (
  val data: Seq[LabeledObject[X, Y]],
  val dissolveFunctions: DissolveFunctions[X, Y],
  val solverOptions: SolverOptions[X, Y],
  val testData: Seq[LabeledObject[X, Y]]) {

  // Constructor without test data
  def this(
    data: Seq[LabeledObject[X, Y]],
    dissolveFunctions: DissolveFunctions[X, Y],
    solverOptions: SolverOptions[X, Y]) = this(data, dissolveFunctions, solverOptions, null)

  val roundLimit = solverOptions.roundLimit
  val lambda = solverOptions.lambda
  val debugOn: Boolean = solverOptions.debug

  val debugSb: StringBuilder = new StringBuilder()

  val maxOracle = dissolveFunctions.oracleFn _
  val phi = dissolveFunctions.featureFn _
  val lossFn = dissolveFunctions.lossFn _
  // Number of dimensions of \phi(x, y)
  val ndims: Int = phi(data(0).pattern, data(0).label).size

  val eps: Double = 2.2204E-16

  /**
   * BCFW optimizer
   */
  def optimize()(implicit m: ClassTag[Y]): (StructSVMModel[X, Y], String) = {

    val verboseDebug: Boolean = false
    val debugWeights: Boolean = false

    /* Initialization */
    var k: Integer = 0
    val n: Int = data.length
    val d: Int = phi(data(0).pattern, data(0).label).size
    // Use first example to determine dimension of w
    val model: StructSVMModel[X, Y] = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(d), dissolveFunctions)
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
    val debugModel: StructSVMModel[X, Y] = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(ndims), dissolveFunctions)

    if (solverOptions.debug) {
      if (solverOptions.testData != null)
        debugSb ++= "round,time,iter,primal,dual,gap,train_error,test_error\n"
      else
        debugSb ++= "round,time,iter,primal,dual,gap,train_error\n"
    }
    val startTime = System.currentTimeMillis()

    if (debugOn) {
      println("Beginning training of %d data points in %d passes with lambda=%f".format(n, roundLimit, lambda))
    }

    // Initialize the cache: Index -> List of precomputed ystar_i's
    var oracleCache = collection.mutable.Map[Int, MutableList[Y]]()

    for (passNum <- 0 until roundLimit) {

      if (verboseDebug) {
        println("wMat before pass: " + model.getWeights()(0 to 10).toDenseVector)
        println("ellmat before pass: " + ellMat(0 to 10))
        println("Ell before pass = " + ell)
      }

      for (dummy <- 0 until n) {
        // 1) Pick example
        val i: Int = dummy
        val pattern: X = data(i).pattern
        val label: Y = data(i).label

        // 2) Solve loss-augmented inference for point i
        // 2.a) If cache is enabled, check if any of the previous ystar_i's for this i can be used
        val bestCachedCandidateForI: Option[Y] =
          if (solverOptions.enableOracleCache && oracleCache.contains(i)) {
            val candidates: Seq[(Double, Int)] =
              oracleCache(i)
                .map(y_i => (((phi(pattern, label) - phi(pattern, y_i)) :* (1 / (n * lambda))),
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
            if (candidates.size >= 1)
              Some(oracleCache(i)(candidates.head._2))
            else None
          } else
            None

        // 2.b) In case cache is disabled or a good contender from cache hasn't been found, call max Oracle
        val ystar_i: Y =
          if (bestCachedCandidateForI.isEmpty) {
            val ystar = maxOracle(model, pattern, label)

            if (solverOptions.enableOracleCache)
              // Add this newly computed ystar to the cache of this i
              oracleCache.update(i, if (solverOptions.oracleCacheSize > 0)
                { oracleCache.getOrElse(i, MutableList[Y]()) :+ ystar }.takeRight(solverOptions.oracleCacheSize)
              else { oracleCache.getOrElse(i, MutableList[Y]()) :+ ystar })
            // kick out oldest if max size reached
            ystar
          } else {
            bestCachedCandidateForI.get
          }

        if (i < 5 && verboseDebug) {
          println("y_star(%d) = %s".format(i, ystar_i))
        }

        // 3) Define the update quantities
        val psi_i: Vector[Double] = phi(pattern, label) - phi(pattern, ystar_i)
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
        if (debugWeights && dummy == (n - 1))
          csvwrite(new File("data/debug/debugWeights/scala-w-%d.csv".format(passNum + 1)),
            { if (solverOptions.doWeightedAveraging) wAvg else model.getWeights }.toDenseVector.toDenseMatrix)

        k = k + 1

        if (debugOn && k >= debugIter) {
          if (solverOptions.doWeightedAveraging) {
            debugModel.updateWeights(wAvg)
            debugModel.updateEll(lAvg)
          } else {
            debugModel.updateWeights(model.getWeights)
            debugModel.updateEll(ell)
          }

          val f = -SolverUtils.objectiveFunction(debugModel.getWeights(), debugModel.getEll(), lambda)
          val gapTup = SolverUtils.dualityGap(data, phi, lossFn, maxOracle, debugModel, lambda)
          val gap = gapTup._1
          val primal = f + gap
          val trainError = SolverUtils.averageLoss(data, dissolveFunctions, debugModel)

          if (verboseDebug) {
            println("wMat after pass: " + model.getWeights()(0 to 10).toDenseVector)
            println("ellmat after pass: " + ellMat(0 to 10))
            println("Ell after pass = " + ell)

            debugSb ++= "# sum(w): %f, ell: %f\n".format(debugModel.getWeights().sum, debugModel.getEll())
          }

          val curTime = (System.currentTimeMillis() - startTime) / 1000.0

          if (solverOptions.testData != null) {
            val testError =
              if (solverOptions.testData.isDefined)
                SolverUtils.averageLoss(solverOptions.testData.get, dissolveFunctions, debugModel)
              else
                0.00
            println("Pass %d Iteration %d, SVM primal = %f, SVM dual = %f, Duality gap = %f, Train error = %f, Test error = %f"
              .format(passNum + 1, k, primal, f, gap, trainError, testError))
            if (solverOptions.debug)
              debugSb ++= "%d,%f,%d,%s,%s,%s,%f,%f\n".format(passNum + 1, curTime, k, primal.toString(), f.toString(), gap.toString(), trainError, testError)
          } else {
            println("Pass %d Iteration %d, SVM primal = %f, SVM dual = %f, Duality gap = %f, Train error = %f"
              .format(passNum + 1, k, primal, f, gap, trainError))
            if (solverOptions.debug)
              debugSb ++= "%d,%f,%d,%s,%s,%s,%f\n".format(passNum + 1, curTime, k, primal.toString(), f.toString(), gap.toString(), trainError)
          }

          debugIter = min(debugIter + n, ceil(debugIter * (1 + solverOptions.debugMultiplier / 100)))
        }

      }

    }

    if (solverOptions.doWeightedAveraging) {
      model.updateWeights(wAvg)
      model.updateEll(lAvg)
    } else {
      model.updateEll(ell)
    }

    return (model, debugSb.toString())
  }

}