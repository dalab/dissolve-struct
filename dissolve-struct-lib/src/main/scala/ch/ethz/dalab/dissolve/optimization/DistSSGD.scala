package ch.ethz.dalab.dissolve.optimization

import scala.reflect.ClassTag
import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import breeze.linalg.DenseVector
import breeze.linalg.InjectNumericOps
import breeze.linalg.SparseVector
import breeze.linalg.Vector
import breeze.linalg.norm
import ch.ethz.dalab.dissolve.classification.MutableWeightsEll
import ch.ethz.dalab.dissolve.classification.Types.Index
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.classification.MutableWeightsEll

class DistSSGD[X, Y](
  dissolveFunctions: DissolveFunctions[X, Y],
  roundLimit: Int = 200,
  doWeightedAveraging: Boolean = true,
  beta: Double = 1.0,
  eta: Double = 1.0,
  timeBudget: Int = Integer.MAX_VALUE,
  debug: Boolean = false,
  debugMultiplier: Int = 100,
  debugOutPath: String = "debug-%d.csv".format(System.currentTimeMillis()),
  lambda: Double = 0.01,
  samplePerRound: Double = 0.5,
  sparse: Boolean = false,
  checkpointFreq: Int = 50)
    extends Serializable with DistributedSolver[X, Y] {

  case class SSGDRoundEvaluation(roundNum: Int,
                                 passNum: Double,
                                 elapsedTime: Double,
                                 wallTime: Double,
                                 primal: Double,
                                 trainError: Double,
                                 testError: Double,
                                 trainStructHingeLoss: Double,
                                 testStructHingeLoss: Double,
                                 w_t_norm: Double,
                                 w_update_norm: Double,
                                 cos_w_update: Double) {

    override def toString(): String = {

      "%d,%f,%f,%f,%s,%f,%f,%f,%f,%s,%s,%s"
        .format(roundNum, passNum, elapsedTime, wallTime, primal.toString(),
          trainError, testError, trainStructHingeLoss, testStructHingeLoss,
          w_t_norm, w_update_norm, cos_w_update)
    }
  }

  val header: String =
    "round,pass,time,wall_time,primal,train_error,test_error,train_loss,test_loss,w_t,w_diff,w_cos\n"

  val EPS: Double = 2.2204E-16

  // Beyond `DEBUG_THRESH` rounds, debug calculations occur every `DEBUG_STEP`-th round
  val DEBUG_THRESH: Int = 100
  val DEBUG_STEP: Int = 50
  var nextDebugRound: Int = 1

  // Amount of time (in ms) spent in debug operation,
  // i.e, getting gap. errors, etc.
  var evaluateModelTimeMillis: Long = 0

  /**
   * This runs on the Master node, and each round triggers a map-reduce job on the workers
   */
  def train(data: RDD[LabeledObject[X, Y]],
            testData: Option[RDD[LabeledObject[X, Y]]])(implicit m: ClassTag[Y]): Vector[Double] = {

    val startTime = System.currentTimeMillis()

    val sc = data.context
    val numPartitions: Int = data.partitions.size

    val debugSb: StringBuilder = new StringBuilder()

    /**
     *  Create four RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     *  3. indexedCacheRDD (Index, BoundedCacheList)
     *  4. indexedLocalProcessedData (Index, LocallyProcessedData)
     *  all of which are partitioned similarly
     */

    /**
     * 1. Training/Test Data RDDs
     */
    // Training
    val indexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      data
        .zipWithIndex
        .map {
          case (labeledObject, idx) =>
            (idx.toInt, labeledObject)
        }
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()
    // Test
    val indexedTestDataRDD =
      if (testData.isDefined)
        Some(
          {
            testData.get
              .zipWithIndex
              .map {
                case (labeledObject, idx) =>
                  (idx.toInt, labeledObject)
              }
              .partitionBy(new HashPartitioner(numPartitions))
              .cache()
          })
      else
        None

    /**
     * 1.b. Obtain and set parameters from training data
     */
    val dataSize = indexedTrainDataRDD.count().toInt
    val samplePoint = indexedTrainDataRDD.take(1)(0)._2
    val testDataSize =
      if (indexedTestDataRDD.isDefined)
        indexedTestDataRDD.get.count().toInt
      else
        0
    val d: Int = dissolveFunctions.featureFn(samplePoint.pattern, samplePoint.label).size
    val kAccum = DenseVector.zeros[Int](numPartitions)

    data.unpersist()
    if (testData.isDefined) testData.get.unpersist()

    // Let the initial model contain zeros for all weights
    // Global model uses Dense Vectors by default
    var globalModel: MutableWeightsEll = new MutableWeightsEll(
      if (sparse) SparseVector.zeros(d) else DenseVector.zeros(d),
      0.0)
    var globalModelWeightedAverage: MutableWeightsEll = new MutableWeightsEll(
      if (sparse) SparseVector.zeros(d) else DenseVector.zeros(d),
      0.0)

    def getElapsedTimeSecs(): Double = ((System.currentTimeMillis() - startTime) / 1000.0)

    def getLatestModel(): MutableWeightsEll = {
      if (doWeightedAveraging)
        globalModelWeightedAverage.clone()
      else
        globalModel.clone()
    }

    def evaluateModel(model: MutableWeightsEll, roundNum: Int = 0,
                      w_t_norm: Double, w_update_norm: Double, cos_w_update: Double): SSGDRoundEvaluation = {

      val startEvaluateTime = System.currentTimeMillis()

      val trainingDataRDD = indexedTrainDataRDD.values

      val primal =
        SolverUtils.primalObjective(trainingDataRDD, dataSize, dissolveFunctions, model, lambda)._1

      val (trainError, trainHingeLoss) = SolverUtils.averageLoss(trainingDataRDD, dissolveFunctions, model, dataSize)
      val (testError, testHingeLoss) =
        if (indexedTestDataRDD.isDefined)
          SolverUtils.averageLoss(indexedTestDataRDD.get.values, dissolveFunctions, model, testDataSize)
        else
          (0.0, 0.0)

      val endEvaluateTime = System.currentTimeMillis()

      evaluateModelTimeMillis += (endEvaluateTime - startEvaluateTime)

      val elapsedTime = getElapsedTimeSecs()

      val wallTime = elapsedTime - (evaluateModelTimeMillis / 1000.0)

      println("[%.3f] WallTime = %.3f, Round = %d, Primal = %s, TrainLoss = %f, TestLoss = %f, TrainSHLoss = %f, TestSHLoss = %f"
        .format(elapsedTime, wallTime, roundNum, primal.toString(), trainError, testError, trainHingeLoss, testHingeLoss))

      val roundEval = SSGDRoundEvaluation(roundNum, roundNum * samplePerRound, elapsedTime, wallTime, primal,
        trainError, testError, trainHingeLoss, testHingeLoss,
        w_t_norm, w_update_norm, cos_w_update)

      roundEval
    }

    /**
     * ==== Begin Training rounds ====
     */
    (1 to roundLimit).toStream
      .takeWhile {
        roundNum =>

          val timeLimitExceeded = (getElapsedTimeSecs() / 60.0) > timeBudget

          val continueExecution = !timeLimitExceeded

          if (debug && (!continueExecution || roundNum == 1)) {
            // Force evaluation of model in 2 cases - Before beginning the very first round, and after the last round
            debugSb ++= evaluateModel(getLatestModel(), if (roundNum == 1) 0 else roundNum, Double.NaN, Double.NaN, Double.NaN) + "\n"
          }

          continueExecution
      }
      .foreach {
        roundNum =>

          val deltaInfoRDD = indexedTrainDataRDD
            .sample(withReplacement = false, fraction = samplePerRound)
            .mapPartitionsWithIndex((idx, dataIterator) => mapper((idx, numPartitions), dataIterator, globalModel, null), true)
            .cache()

          // Trigger an action
          val newKPerPartition = deltaInfoRDD.map {
            case (partitionIdx, newWeights, newK) =>
              partitionIdx -> newK
          }.collect()

          newKPerPartition.foreach {
            case (partitionIdx, newK) =>
              kAccum(partitionIdx) = newK
          }

          val sumDeltaWeightsEll = deltaInfoRDD.map {
            case (partitionIdx, newWeights, newK) =>
              newWeights
          }.reduce(_ + _)

          val oldWeights = globalModel.getWeights()
          val newWeights = globalModel.getWeights() - ((beta / numPartitions) * sumDeltaWeightsEll.getWeights())

          globalModel.setWeights(newWeights)

          deltaInfoRDD.unpersist()

          val w_t = oldWeights
          val w_tp1 = newWeights

          // || w_t ||
          val w_t_norm = norm(w_t, 2)

          // || w_{t+1} - w_t} ||
          val w_diff_norm = norm(w_tp1 - w_t, 2)

          // cos( w_t , w_{t-1} )
          val cos_w = (w_t dot w_tp1) / (norm(w_t, 2) * norm(w_tp1, 2))

          // Obtain duality gap after each communication round
          val debugModel: MutableWeightsEll =
            if (doWeightedAveraging)
              globalModelWeightedAverage.clone()
            else globalModel.clone()

          // Is criteria for debugging met?
          val doDebugCalc: Boolean =
            if (debugMultiplier == 1) {
              true
            } else if (roundNum <= DEBUG_THRESH && roundNum == nextDebugRound) {
              nextDebugRound = nextDebugRound * debugMultiplier
              true
            } else if (roundNum > DEBUG_THRESH && roundNum % DEBUG_STEP == 0) {
              nextDebugRound += DEBUG_STEP
              true
            } else
              false

          val roundEvaluation =
            if (debug && doDebugCalc) {
              evaluateModel(debugModel, roundNum, w_t_norm, w_diff_norm, cos_w)
            } else {
              // If debug flag isn't on, perform calculations that don't trigger a shuffle
              val elapsedTime = getElapsedTimeSecs()

              val wallTime = elapsedTime - (evaluateModelTimeMillis / 1000.0)

              SSGDRoundEvaluation(roundNum, roundNum * samplePerRound,
                elapsedTime, wallTime,
                Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN,
                w_t_norm, w_diff_norm, cos_w)
            }

          debugSb ++= roundEvaluation + "\n"

      }

    globalModel.getWeights()
  }

  /**
   * @returns An iterator triplet (partitionIdx, newW, newK)
   */
  def mapper(partitionInfo: (Int, Int), // (partitionIdx, numPartitions)
             dataIterator: Iterator[(Index, LabeledObject[X, Y])],
             weightsObj: MutableWeightsEll,
             kAccum: Vector[Int]): Iterator[(Int, MutableWeightsEll, Int)] = {

    val (partitionIdx, numPartitions) = partitionInfo
    var k = kAccum(partitionIdx)

    val w = weightsObj.getWeights()
    val psiSum =
      dataIterator
        .map {
          case (idx, lo) =>
            val xi: X = lo.pattern
            val yi: Y = lo.label
            val ystar: Y = dissolveFunctions.oracleFn(w, xi, yi)

            k += 1

            val psi =
              dissolveFunctions.featureFn(xi, yi) - dissolveFunctions.featureFn(xi, ystar)

            psi
        }
        .reduce(_ + _)

    val p = (lambda * w) - psiSum
    val gamma_t: Double = 1.0 / (eta * (k + 1.0))
    val deltaw = p * gamma_t

    val deltawObj = new MutableWeightsEll(deltaw, 0.0)

    List((partitionIdx, deltawObj, k)).toIterator
  }

}