/**
 *
 */
package ch.ethz.dalab.dissolve.optimization

import java.io.File
import java.io.FileWriter

import scala.collection.mutable.MutableList
import scala.reflect.ClassTag

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.linalg.csvwrite
import breeze.linalg.max
import breeze.linalg.min
import breeze.numerics.ceil
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.regression.LabeledObject

/**
 * Train a structured SVM using the primal dual Block-Coordinate Frank-Wolfe solver (BCFW).
 *
 * The implementation here is single machine, not distributed. See DBCFWSolver... for the
 * distributed version.
 *
 * @param <X> type for the data examples
 * @param <Y> type for the labels of each example
 */
class LocalBCFW[X, Y](
    dissolveFunctions: DissolveFunctions[X, Y],
    numPasses: Int = 200,
    doLineSearch: Boolean = true,
    doWeightedAveraging: Boolean = true,
    timeBudget: Int = Integer.MAX_VALUE,
    debug: Boolean = false,
    debugMultiplier: Int = 100,
    debugOutPath: String = "debug-%d.csv".format(System.currentTimeMillis()),
    randSeed: Long = 1,
    randomSampling: Boolean = false,
    lambda: Double = 0.01,
    gapThreshold: Double = 0.1,
    gapCheck: Int = 0,
    enableOracleCache: Boolean = false,
    oracleCacheSize: Int = 10) extends LocalSolver[X, Y] {

  val debugSb: StringBuilder = new StringBuilder()

  val maxOracle = dissolveFunctions.oracleFn _
  val phi = dissolveFunctions.featureFn _
  val lossFn = dissolveFunctions.lossFn _

  val eps: Double = 2.2204E-16

  val solverParamsStr: String = {

    val sb: StringBuilder = new StringBuilder()

    sb ++= "# numPasses=%s\n".format(numPasses)
    sb ++= "# doLineSearch=%s\n".format(doLineSearch)
    sb ++= "# doWeightedAveraging=%s\n".format(doWeightedAveraging)
    sb ++= "# timeBudget=%s\n".format(timeBudget)

    sb ++= "# debug=%s\n".format(debug)
    sb ++= "# debugMultiplier=%s\n".format(debugMultiplier)
    sb ++= "# debugOutPath=%s\n".format(debugOutPath)

    sb ++= "# randSeed=%s\n".format(randSeed)
    sb ++= "# randomSampling=%s\n".format(randomSampling)

    sb ++= "# lambda=%s\n".format(lambda)

    sb ++= "# gapThreshold=%s\n".format(gapThreshold)
    sb ++= "# gapCheck=%s\n".format(gapCheck)

    sb ++= "# enableOracleCache=%s\n".format(enableOracleCache)
    sb ++= "# oracleCacheSize=%s\n".format(oracleCacheSize)

    sb.toString()
  }

  /**
   * BCFW optimizer
   */
  def train(data: Seq[LabeledObject[X, Y]],
            testData: Option[Seq[LabeledObject[X, Y]]])(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = {

    util.Random.setSeed(randSeed)

    // Number of dimensions of \phi(x, y)
    val ndims: Int = phi(data(0).pattern, data(0).label).size

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
      if (doWeightedAveraging)
        DenseVector.zeros(d)
      else null
    var lAvg: Double = 0.0

    val debugMultiplierCor = if (debugMultiplier == 0) 100 else debugMultiplier
    var debugIter = if (debugMultiplier == 0) n else 1
    val debugModel: StructSVMModel[X, Y] = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(ndims), dissolveFunctions)

    if (debug) {
      if (testData != null)
        debugSb ++= "round,time,iter,primal,dual,gap,train_error,test_error\n"
      else
        debugSb ++= "round,time,iter,primal,dual,gap,train_error\n"
    }
    val startTimeMillis = System.currentTimeMillis()

    if (debug) {
      println("Beginning training of %d data points in %d passes with lambda=%f".format(n, numPasses, lambda))
    }

    // Initialize the cache: Index -> List of precomputed ystar_i's
    var oracleCache = collection.mutable.Map[Int, MutableList[Y]]()

    // Serves as a flag to determine when to end solver iterations
    // This could be flipped either due to: gapCheck or timeBudget
    var terminateIter: Boolean = false

    (0 until numPasses).toStream
      .takeWhile(_ => !terminateIter)
      .foreach {

        passNum =>
          val idxSeq = (0 until n).toList
          val perm = util.Random.shuffle(idxSeq)

          (0 until n).toStream
            .takeWhile(_ => !terminateIter)
            .foreach {

              dummy =>
                // 1) Pick example
                val i: Int = if (randomSampling) util.Random.nextInt(n) else perm(dummy)
                val pattern: X = data(i).pattern
                val label: Y = data(i).label

                // 2) Solve loss-augmented inference for point i
                // 2.a) If cache is enabled, check if any of the previous ystar_i's for this i can be used
                val bestCachedCandidateForI: Option[Y] =
                  if (enableOracleCache && oracleCache.contains(i)) {
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

                    if (enableOracleCache)
                      // Add this newly computed ystar to the cache of this i
                      oracleCache.update(i, if (oracleCacheSize > 0)
                        { oracleCache.getOrElse(i, MutableList[Y]()) :+ ystar }.takeRight(oracleCacheSize)
                      else { oracleCache.getOrElse(i, MutableList[Y]()) :+ ystar })
                    // kick out oldest if max size reached
                    ystar
                  } else {
                    bestCachedCandidateForI.get
                  }

                // 3) Define the update quantities
                val psi_i: Vector[Double] = phi(pattern, label) - phi(pattern, ystar_i)
                val w_s: Vector[Double] = psi_i :* (1 / (n * lambda))
                val loss_i: Double = lossFn(label, ystar_i)
                val ell_s: Double = (1.0 / n) * loss_i

                // 4) Get step-size gamma
                val gamma: Double =
                  if (doLineSearch) {
                    val gamma_opt = (model.getWeights().t * (wMat(::, i) - w_s) - ((ellMat(i) - ell_s) * (1 / lambda))) /
                      ((wMat(::, i) - w_s).t * (wMat(::, i) - w_s) + eps)
                    max(0.0, min(1.0, gamma_opt))
                  } else {
                    (2.0 * n) / (k + 2.0 * n)
                  }

                // 5, 6, 7, 8) Update the weights of the model
                val tempWeights1: Vector[Double] = model.getWeights() - wMat(::, i)
                model.setWeights(tempWeights1)
                wMat(::, i) := (wMat(::, i) * (1.0 - gamma)) + (w_s * gamma)
                val tempWeights2: Vector[Double] = model.getWeights() + wMat(::, i)
                model.setWeights(tempWeights2)

                ell = ell - ellMat(i)
                ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma)
                ell = ell + ellMat(i)

                // 9) Optionally update the weighted average
                if (doWeightedAveraging) {
                  val rho: Double = 2.0 / (k + 2.0)
                  wAvg = (wAvg * (1.0 - rho)) + (model.getWeights * rho)
                  lAvg = (lAvg * (1.0 - rho)) + (ell * rho)
                }

                /**
                 * DEBUG/TEST code
                 */
                k = k + 1

                if (debug && k >= debugIter) {
                  if (doWeightedAveraging) {
                    debugModel.setWeights(wAvg)
                    debugModel.setEll(lAvg)
                  } else {
                    debugModel.setWeights(model.getWeights)
                    debugModel.setEll(ell)
                  }

                  val f = -SolverUtils.objectiveFunction(debugModel.getWeights(), debugModel.getEll(), lambda)
                  val gapTup = SolverUtils.dualityGap(data, phi, lossFn, maxOracle, debugModel, lambda)
                  val gap = gapTup._1
                  val primal = f + gap

                  val trainError = SolverUtils.averageLoss(data, dissolveFunctions, debugModel)
                  val testError =
                    if (testData.isDefined)
                      SolverUtils.averageLoss(testData.get, dissolveFunctions, debugModel)
                    else
                      (0.00, 0.00)

                  val curTime = (System.currentTimeMillis() - startTimeMillis) / 1000.0

                  println("Pass %d Iteration %d, SVM primal = %f, SVM dual = %f, Duality gap = %f, Train error = %f, Test error = %f"
                    .format(passNum + 1, k, primal, f, gap, trainError._1, testError._1))

                  debugSb ++= "%d,%f,%d,%s,%s,%s,%f,%f\n"
                    .format(passNum + 1, curTime, k,
                      primal, f, gap,
                      trainError._1, testError._1)

                  debugIter = min(debugIter + n, ceil(debugIter * (1 + debugMultiplierCor / 100)))
                }

                // Check if time budget is exceeded
                val elapsedTimeMillis = System.currentTimeMillis() - startTimeMillis
                val elapsedTimeMins = elapsedTimeMillis / (1000.0 * 60.0)
                if (elapsedTimeMins >= timeBudget) {
                  println("Time budget exceeded. Stopping.")
                  println("Elapsed Time : %s mins. TimeBudget : %s mins".format(elapsedTimeMins, timeBudget))
                  terminateIter = true
                }

            }

          // Check if gap criterion is met
          if (gapCheck > 0 && (passNum + 1) % gapCheck == 0) {

            if (doWeightedAveraging) {
              debugModel.setWeights(wAvg)
              debugModel.setEll(lAvg)
            } else {
              debugModel.setWeights(model.getWeights)
              debugModel.setEll(ell)
            }

            val gapTup = SolverUtils.dualityGap(data, phi, lossFn, maxOracle, debugModel, lambda)
            val gap = gapTup._1

            if (gap <= gapThreshold) {
              println("Duality gap requirement satisfied. Stopping.")
              println("Current gap: %s. Gap threshold = %s".format(gap, gapThreshold))
              terminateIter = true
            }

          }

      }

    if (doWeightedAveraging) {
      model.setWeights(wAvg)
      model.setEll(lAvg)
    } else {
      model.setEll(ell)
    }

    /**
     * Write debug stats
     */
    // Create intermediate directories if necessary
    val outFile = new java.io.File(debugOutPath)
    val outDir = outFile.getAbsoluteFile().getParentFile()
    if (!outDir.exists()) {
      println("Directory %s does not exist. Creating required path.")
      outDir.mkdirs()
    }

    // Dump debug information into a file
    val fw = new FileWriter(outFile)
    // Write the current parameters being used
    fw.write(solverParamsStr)
    fw.write("\n")

    // Write values noted from the run
    fw.write(debugSb.toString())
    fw.close()

    print(debugSb)

    return model
  }

}