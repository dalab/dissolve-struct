package ch.ethz.dalab.dissolve.optimization

import java.io.File
import java.io.PrintWriter
import breeze.linalg._
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.linalg.csvwrite
import breeze.numerics._
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.regression.LabeledObject
import scala.collection.mutable.MutableList
import scala.reflect.ClassTag
import java.io.FileWriter

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
class LocalSSGD[X, Y](
    dissolveFunctions: DissolveFunctions[X, Y],
    numPasses: Int = 200,
    doWeightedAveraging: Boolean = true,
    timeBudget: Int = Integer.MAX_VALUE,
    debug: Boolean = false,
    debugMultiplier: Int = 100,
    debugOutPath: String = "debug-%d.csv".format(System.currentTimeMillis()),
    randSeed: Long = 1,
    randomSampling: Boolean = false,
    lambda: Double = 0.01,
    eta: Double = 1.0) extends LocalSolver[X, Y] {

  val debugSb: StringBuilder = new StringBuilder()

  val maxOracle = dissolveFunctions.oracleFn _
  val phi = dissolveFunctions.featureFn _
  val lossFn = dissolveFunctions.lossFn _

  val eps: Double = 2.2204E-16

  val solverParamsStr: String = {

    val sb: StringBuilder = new StringBuilder()

    sb ++= "# numPasses=%s\n".format(numPasses)
    sb ++= "# doWeightedAveraging=%s\n".format(doWeightedAveraging)
    sb ++= "# timeBudget=%s\n".format(timeBudget)

    sb ++= "# debug=%s\n".format(debug)
    sb ++= "# debugMultiplier=%s\n".format(debugMultiplier)
    sb ++= "# debugOutPath=%s\n".format(debugOutPath)

    sb ++= "# randSeed=%s\n".format(randSeed)
    sb ++= "# randomSampling=%s\n".format(randomSampling)

    sb ++= "# lambda=%s\n".format(lambda)
    sb ++= "# eta=%s\n".format(eta)

    sb.toString()
  }

  /**
   * BCFW optimizer
   */
  def train(data: Seq[LabeledObject[X, Y]],
            testData: Option[Seq[LabeledObject[X, Y]]])(implicit m: ClassTag[Y]): Vector[Double] = {

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

    // Initialization in case of Weighted Averaging
    var wAvg: DenseVector[Double] =
      if (doWeightedAveraging)
        DenseVector.zeros(d)
      else null

    val debugMultiplierCor = if (debugMultiplier == 0) 100 else debugMultiplier
    var debugIter = if (debugMultiplier == 0) n else 1
    val debugModel: StructSVMModel[X, Y] = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(ndims), dissolveFunctions)

    if (debug) {
      debugSb ++= "round,time,iter,primal,train_error\n"
    }
    val startTimeMillis = System.currentTimeMillis()

    if (debug) {
      println("Beginning training of %d data points in %d passes with lambda=%f".format(n, numPasses, lambda))
    }

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
                val ystar_i: Y = maxOracle(model.getWeights(), pattern, label)

                // 3) Get the subgradient
                val psi_i: Vector[Double] = (phi(pattern, label) - phi(pattern, ystar_i))
                val w_s: Vector[Double] = psi_i :* (1 / (n * lambda))

                // 4) Get step-size gamma
                val gamma: Double = 1.0 / (eta * (k + 1.0))

                // 5) Update the weights of the model
                val newWeights: Vector[Double] = (model.getWeights() :* (1 - gamma)) + (w_s :* (gamma * n))
                model.setWeights(newWeights)

                // 9) Optionally update the weighted average
                if (doWeightedAveraging) {
                  val rho: Double = 2.0 / (k + 2.0)
                  wAvg = (wAvg * (1.0 - rho)) + (model.getWeights * rho)
                }

                /**
                 * DEBUG/TEST code
                 */
                k = k + 1

                if (debug && k >= debugIter) {
                  if (doWeightedAveraging) {
                    debugModel.setWeights(wAvg)
                  } else {
                    debugModel.setWeights(model.getWeights)
                  }

                  val primal = SolverUtils.primalObjective(data, dissolveFunctions, debugModel, lambda)

                  val trainError = SolverUtils.averageLoss(data, dissolveFunctions, debugModel)
                  val testError =
                    if (testData.isDefined)
                      SolverUtils.averageLoss(testData.get, dissolveFunctions, debugModel)
                    else
                      (0.00, 0.00)

                  val curTime = (System.currentTimeMillis() - startTimeMillis) / 1000.0

                  println("Pass %d Iteration %d, SVM primal = %f, Train error = %f, Test error = %f"
                    .format(passNum + 1, k, primal, trainError._1, testError._1))

                  debugSb ++= "%d,%f,%d,%s,%f,%f\n"
                    .format(passNum + 1, curTime, k,
                      primal, trainError._1, testError._1)

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

      }

    if (doWeightedAveraging)
      model.setWeights(wAvg)

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

    return model.getWeights()
  }

}