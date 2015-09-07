package ch.ethz.dalab.dissolve.optimization

import scala.collection.mutable.MutableList
import scala.reflect.ClassTag
import org.apache.spark.HashPartitioner
import org.apache.spark.SparkContext.rddToPairRDDFunctions
import org.apache.spark.rdd.RDD
import breeze.linalg.DenseVector
import breeze.linalg.SparseVector
import breeze.linalg.Vector
import breeze.linalg.max
import breeze.linalg.min
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.classification.Types.BoundedCacheList
import ch.ethz.dalab.dissolve.classification.Types.Index
import ch.ethz.dalab.dissolve.classification.Types.PrimalInfo
import ch.ethz.dalab.dissolve.regression.LabeledObject
import scala.collection.mutable.PriorityQueue
import breeze.linalg.norm
import org.apache.log4j.Logger

object LAdap extends Serializable {
  @transient lazy val log =
    Logger.getLogger(getClass.getName)
}

/**
 * Train a structured SVM using the actual distributed dissolve^struct solver.
 * This uses primal dual Block-Coordinate Frank-Wolfe solver (BCFW), distributed
 * via the CoCoA framework (Communication-Efficient Distributed Dual Coordinate Ascent)
 *
 * @param <X> type for the data examples
 * @param <Y> type for the labels of each example
 */
class DBCFWSolverTuned[X, Y](
    val data: RDD[LabeledObject[X, Y]],
    val dissolveFunctions: DissolveFunctions[X, Y],
    val solverOptions: SolverOptions[X, Y],
    val miniBatchEnabled: Boolean = false) extends Serializable {

  val ENABLE_PERF_METRICS: Boolean = false

  def time[R](block: => R, blockName: String = ""): R = {
    if (ENABLE_PERF_METRICS) {
      val t0 = System.currentTimeMillis()
      val result = block // call-by-name
      val t1 = System.currentTimeMillis()
      println("%25s %d ms".format(blockName, (t1 - t0)))
      result
    } else block
  }

  /**
   * Some case classes to make code more readable
   */

  case class HelperFunctions[X, Y](featureFn: (X, Y) => Vector[Double],
                                   lossFn: (Y, Y) => Double,
                                   oracleFn: (StructSVMModel[X, Y], X, Y) => Y,
                                   oracleStreamFn: (StructSVMModel[X, Y], X, Y) => Stream[Y],
                                   predictFn: (StructSVMModel[X, Y], X) => Y,
                                   fineOracleFn: (StructSVMModel[X, Y], X, Y) => Y,
                                   xid: (X) => String)

  // Input to the mapper: idx -> DataShard
  case class InputDataShard[X, Y](labeledObject: LabeledObject[X, Y],
                                  primalInfo: PrimalInfo,
                                  cache: Option[BoundedCacheList[Y]])

  // Output of the mapper: idx -> ProcessedDataShard
  case class ProcessedDataShard[X, Y](primalInfo: PrimalInfo,
                                      cache: Option[BoundedCacheList[Y]],
                                      localSummary: Option[LocalSummary[X, Y]])

  case class LocalSummary[X, Y](deltaLocalModel: StructSVMModel[X, Y],
                                deltaLocalK: Vector[Int],
                                deltaLocalAveragedModel: StructSVMModel[X, Y])

  // Experimental data
  case class RoundEvaluation(roundNum: Int,
                             elapsedTime: Double,
                             wallTime: Double,
                             primal: Double,
                             dual: Double,
                             dualityGap: Double,
                             trainError: Double,
                             testError: Double,
                             trainStructHingeLoss: Double,
                             testStructHingeLoss: Double,
                             w_t_norm: Double,
                             w_update_norm: Double,
                             cos_w_update: Double,
                             perClassAccuracyTrain: Array[Double],
                             globalAccuracyTrain: Double,
                             perClassAccuracyTest: Array[Double],
                             globalAccuracyTest: Double) {

    override def toString(): String = {

      // Average per-class accuracy WITHOUT incomplete labels
      val numClasses = dissolveFunctions.numClasses()
      /**
       * Train
       */
      assert(perClassAccuracyTrain.size == numClasses)
      // Assume background label is the last class
      val candidateLabelsTrain = perClassAccuracyTrain.dropRight(1)
      // Compute only over non-zero candidates. (To handle unencountered classes)
      val nonZeroTrainCandidates = candidateLabelsTrain.filter(_ > 0.0)
      val averagePerClassAccuracyTrain =
        nonZeroTrainCandidates.sum / nonZeroTrainCandidates.size

      // Format: <class 0>, <class 1>, ... , <average>, <global> 
      val perClassAccuracyStringTrain =
        perClassAccuracyTrain
          .foldLeft("") {
            (accum, next) =>
              "%s,%f".format(accum, next)
          }.drop(1) // Drop the first comma, when accum is empty
      val accuracyStringTrain = "%s,%f,%f".format(perClassAccuracyStringTrain,
        averagePerClassAccuracyTrain,
        globalAccuracyTrain)

      /**
       * Test
       */
      assert(perClassAccuracyTest.size == numClasses)
      // Assume background label is the last class
      val candidateLabelsTest = perClassAccuracyTest.dropRight(1)
      // Compute only over non-zero candidates. (To handle unencountered classes)
      val nonZeroTestCandidates = candidateLabelsTest.filter(_ > 0.0)
      val averagePerClassAccuracyTest =
        nonZeroTestCandidates.sum / nonZeroTestCandidates.size

      // Format: <class 0>, <class 1>, ... , <average>, <global> 
      val perClassAccuracyStringTest =
        perClassAccuracyTest
          .foldLeft("") {
            (accum, next) =>
              "%s,%f".format(accum, next)
          }.drop(1) // Drop the first comma, when accum is empty
      val accuracyStringTest = "%s,%f,%f".format(perClassAccuracyStringTest,
        averagePerClassAccuracyTest,
        globalAccuracyTest)

      "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%s,%s"
        .format(roundNum, elapsedTime, wallTime, primal, dual, dualityGap,
          trainError, testError, trainStructHingeLoss, testStructHingeLoss,
          w_t_norm, w_update_norm, cos_w_update,
          accuracyStringTrain, accuracyStringTest)
    }
  }

  val EPS: Double = 2.2204E-16

  // Beyond `DEBUG_THRESH` rounds, debug calculations occur every `DEBUG_STEP`-th round
  val DEBUG_THRESH: Int = 100
  val DEBUG_STEP: Int = 50
  var nextDebugRound: Int = 1

  val header: String = {

    def getAccuracyHeaderWithPrefix(prefix: String): String = {
      val accuracyHeader = new StringBuilder()

      // Per class headers
      for (i <- 0 until dissolveFunctions.numClasses()) {
        accuracyHeader ++= "%s_%d,".format(prefix, i)
      }

      // Average Per-class accuracy
      accuracyHeader ++= "%s_average,".format(prefix)

      // Global
      accuracyHeader ++= "%s_global".format(prefix)

      accuracyHeader.toString()
    }

    "round,time,wall_time,primal,dual,gap,train_error,test_error,train_loss,test_loss,w_t,w_diff,w_cos,%s,%s\n"
      .format(getAccuracyHeaderWithPrefix("train"), getAccuracyHeaderWithPrefix("test"))
  }

  /**
   * This runs on the Master node, and each round triggers a map-reduce job on the workers
   */
  def optimize()(implicit m: ClassTag[Y]): (StructSVMModel[X, Y], String) = {

    val startTime = System.currentTimeMillis()

    val sc = data.context

    val debugSb: StringBuilder = new StringBuilder()

    val samplePoint = data.first()
    val dataSize = data.count().toInt
    val testDataSize = if (solverOptions.testDataRDD.isDefined) solverOptions.testDataRDD.get.count().toInt else 0

    val verboseDebug: Boolean = false

    val d: Int = dissolveFunctions.featureFn(samplePoint.pattern, samplePoint.label).size
    // Let the initial model contain zeros for all weights
    // Global model uses Dense Vectors by default
    var globalModel: StructSVMModel[X, Y] = new StructSVMModel[X, Y](DenseVector.zeros(d), 0.0,
      DenseVector.zeros(d), dissolveFunctions, solverOptions.numClasses)
    var globalModelWeightedAverage: StructSVMModel[X, Y] = new StructSVMModel[X, Y](DenseVector.zeros(d), 0.0,
      DenseVector.zeros(d), dissolveFunctions, solverOptions.numClasses)

    val numPartitions: Int =
      data.partitions.size

    val beta: Double = 1.0

    val helperFunctions: HelperFunctions[X, Y] = HelperFunctions(dissolveFunctions.featureFn,
      dissolveFunctions.lossFn,
      dissolveFunctions.oracleFn,
      dissolveFunctions.oracleCandidateStream,
      dissolveFunctions.predictFn,
      dissolveFunctions.fineOracleFn,
      dissolveFunctions.getImageID)

    /**
     *  Create four RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     *  3. indexedCacheRDD (Index, BoundedCacheList)
     *  4. indexedLocalProcessedData (Index, LocallyProcessedData)
     *  all of which are partitioned similarly
     */

    /*
     * zipWithIndex calls getPartitions. But, partitioning happens in the future.
     * This causes a race condition.
     * See bug: https://issues.apache.org/jira/browse/SPARK-4433
     * 
     * Making do with work-around
     * 
    val indexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      data.zipWithIndex()
        .map {
          case (labeledObject, idx) =>
            (idx.toInt, labeledObject)
        }
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()
    * 
    */

    // The work-around for bug SPARK-4433
    val zippedIndexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      data.zipWithIndex()
        .map {
          case (labeledObject, idx) =>
            (idx.toInt, labeledObject)
        }
    zippedIndexedTrainDataRDD.count()

    val indexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      zippedIndexedTrainDataRDD
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()

    val indexedPrimals: Array[(Index, PrimalInfo)] = (0 until dataSize).toArray.zip(
      // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
      Array.fill(dataSize)((
        if (solverOptions.sparse) // w_i can be either Sparse or Dense 
          SparseVector.zeros[Double](d)
        else
          DenseVector.zeros[Double](d),
        0.0)))
    var indexedPrimalsRDD: RDD[(Index, PrimalInfo)] =
      sc.parallelize(indexedPrimals)
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()

    // For each Primal (i.e, Index), cache a list of Decodings (i.e, Y's)
    // If cache is disabled, add an empty array. This immediately drops the joins later on and saves time in communicating an unnecessary RDD.
    val indexedCache: Array[(Index, BoundedCacheList[Y])] =
      if (solverOptions.enableOracleCache)
        (0 until dataSize).toArray.zip(
          Array.fill(dataSize)(MutableList[Y]()) // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
          )
      else
        Array[(Index, BoundedCacheList[Y])]()
    var indexedCacheRDD: RDD[(Index, BoundedCacheList[Y])] =
      sc.parallelize(indexedCache)
        .partitionBy(new HashPartitioner(numPartitions))
        .cache()

    var indexedLocalProcessedData: RDD[(Index, ProcessedDataShard[X, Y])] = null

    val kAccum = DenseVector.zeros[Int](numPartitions)

    debugSb ++= "# indexedTrainDataRDD.partitions.size=%d\n".format(indexedTrainDataRDD.partitions.size)
    debugSb ++= "# indexedPrimalsRDD.partitions.size=%d\n".format(indexedPrimalsRDD.partitions.size)
    debugSb ++= "# sc.getExecutorStorageStatus.size=%d\n".format(sc.getExecutorStorageStatus.size)

    /**
     * Fix parameters to perform sampling.
     * Use can either specify:
     * a) "count" - Eqv. to 'H' in paper. Number of points to sample in each round.
     * or b) "perc" - Fraction of dataset to sample \in [0.0, 1.0]
     */
    val sampleFrac: Double = {
      if (solverOptions.sample == "frac")
        solverOptions.sampleFrac
      else if (solverOptions.sample == "count")
        math.min(solverOptions.H / dataSize, 1.0)
      else {
        println("[WARNING] %s is not a valid option. Reverting to sampleFrac = 0.5".format(solverOptions.sample))
        0.5
      }
    }

    /**
     * In case of weighted averaging, start off with an all-zero (wAvg, lAvg)
     */
    var wAvg: Vector[Double] =
      if (solverOptions.doWeightedAveraging)
        DenseVector.zeros(d)
      else null
    var lAvg: Double = 0.0

    var iterCount: Int = 0

    // Amount of time (in ms) spent in debug operation,
    // i.e, getting gap. errors, etc.
    var evaluateModelTimeMillis: Long = 0

    def getLatestModel(): StructSVMModel[X, Y] = {
      if (solverOptions.doWeightedAveraging)
        globalModelWeightedAverage.clone()
      else
        globalModel.clone()
    }

    def getLatestGap(): Double = {
      val debugModel: StructSVMModel[X, Y] = getLatestModel()
      val gap = SolverUtils.dualityGap(data, dissolveFunctions, debugModel, solverOptions.lambda, dataSize)
      gap._1
    }

    def evaluateModel(model: StructSVMModel[X, Y], roundNum: Int = 0,
                      w_t_norm: Double, w_update_norm: Double, cos_w_update: Double): RoundEvaluation = {

      val startEvaluateTime = System.currentTimeMillis()

      /*val dual = -SolverUtils.objectiveFunction(model.getWeights(), model.getEll(), solverOptions.lambda)
      val dualityGap = SolverUtils.dualityGap(data, dissolveFunctions, model, solverOptions.lambda, dataSize)._1
      val primal = dual + dualityGap

      val (trainError, trainStructHingeLoss) = SolverUtils.averageLoss(data, dissolveFunctions, model, dataSize)
      val (testError, testStructHingeLoss) =
        if (solverOptions.testDataRDD.isDefined)
          SolverUtils.averageLoss(solverOptions.testDataRDD.get, dissolveFunctions, model, testDataSize)
        else
          (Double.NaN, Double.NaN)*/

      val trainDataEval = SolverUtils.trainDataEval(data, dissolveFunctions, model, solverOptions.lambda, dataSize)
      val dual = -SolverUtils.objectiveFunction(model.getWeights(), model.getEll(), solverOptions.lambda)
      val dualityGap = trainDataEval.gap
      val primal = dual + dualityGap
      val (trainError,
        trainStructHingeLoss,
        trainPerClassAccuracy,
        trainGlobalAccuracy) =
        (trainDataEval.avgDelta,
          trainDataEval.avgHLoss,
          trainDataEval.perClassAccuracy,
          trainDataEval.globalAccuracy)

      val testDataEval = if (solverOptions.testDataRDD.isDefined)
        SolverUtils.trainDataEval(solverOptions.testDataRDD.get, dissolveFunctions, model, solverOptions.lambda, dataSize)
      else null
      val (testError,
        testStructHingeLoss,
        testPerClassAccuracy,
        testGlobalAccuracy) =
        if (solverOptions.testDataRDD.isDefined)
          (testDataEval.avgDelta,
            testDataEval.avgHLoss,
            testDataEval.perClassAccuracy,
            testDataEval.globalAccuracy)
        else
          (Double.NaN,
            Double.NaN,
            Array.fill(dissolveFunctions.numClasses())(Double.NaN),
            0.0)

      val endEvaluateTime = System.currentTimeMillis()
      evaluateModelTimeMillis += (endEvaluateTime - startEvaluateTime)

      val elapsedTime = getElapsedTimeSecs()

      val wallTime = elapsedTime - (evaluateModelTimeMillis / 1000.0)

      println("[%.3f] WallTime = %.3f, Round = %d, Gap = %f, Primal = %f, Dual = %f, TrainLoss = %f, TestLoss = %f, TrainSHLoss = %f, TestSHLoss = %f"
        .format(elapsedTime, wallTime, roundNum, dualityGap, primal, dual, trainError, testError, trainStructHingeLoss, testStructHingeLoss))

      val roundEval = RoundEvaluation(roundNum, elapsedTime, wallTime, primal, dual, dualityGap,
        trainError, testError, trainStructHingeLoss, testStructHingeLoss,
        w_t_norm, w_update_norm, cos_w_update,
        trainPerClassAccuracy, trainGlobalAccuracy, testPerClassAccuracy, testGlobalAccuracy)

      roundEval
    }

    println("Beginning training of %d data points in %d passes with lambda=%f".format(dataSize, solverOptions.roundLimit, solverOptions.lambda))

    debugSb ++= header
    LAdap.log.info("[D] %s,%s,%s,%s,%s,%s,%s,%s".format("expt_name", "ts", "level", "nNodes", "nSupernodes", "filename", "ts_decode", "ts_oracle_init"))
    LAdap.log.info("[G] %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s".format("k", "ts", "level", "filename", "gamma", "w_s", "ell_s", "energy", "gamma_f", "w_s_f", "ell_s_f", "energy_f"))

    def getElapsedTimeSecs(): Double = ((System.currentTimeMillis() - startTime) / 1000.0)

    /**
     * ==== Begin Training rounds ====
     */
    Stream.from(1)
      .takeWhile {
        roundNum =>
          val continueExecution =
            solverOptions.stoppingCriterion match {
              case RoundLimitCriterion => roundNum <= solverOptions.roundLimit
              case TimeLimitCriterion  => getElapsedTimeSecs() < solverOptions.timeLimit
              case GapThresholdCriterion =>
                // Calculating duality gap is really expensive. So, check ever gapCheck rounds
                if (roundNum % solverOptions.gapCheck == 0)
                  getLatestGap() > solverOptions.gapThreshold
                else
                  true
              case _ => throw new Exception("Unrecognized Stopping Criterion")
            }

          if (solverOptions.debug && (!(continueExecution || (roundNum - 1 % solverOptions.debugMultiplier == 0)) || roundNum == 1)) {
            // Force evaluation of model in 2 cases - Before beginning the very first round, and after the last round
            debugSb ++= evaluateModel(getLatestModel(), if (roundNum == 1) 0 else roundNum, Double.NaN, Double.NaN, Double.NaN) + "\n"
          }

          continueExecution
      }
      .foreach {
        roundNum =>

          println("[ROUND %d]".format(roundNum))
          /**
           * Step 1 - Create a joint RDD containing all information of idx -> (data, primals, cache)
           */
          val indexedJointData: RDD[(Index, InputDataShard[X, Y])] =
            indexedTrainDataRDD
              .sample(solverOptions.sampleWithReplacement, sampleFrac)
              .join(indexedPrimalsRDD)
              .leftOuterJoin(indexedCacheRDD)
              .mapValues { // Because mapValues preserves partitioning
                case ((labeledObject, primalInfo), cache) =>
                  InputDataShard(labeledObject, primalInfo, cache)
              }

          /*println("indexedTrainDataRDD = " + indexedTrainDataRDD.count())
          println("indexedJointData.count = " + indexedJointData.count())
          println("indexedPrimalsRDD.count = " + indexedPrimalsRDD.count())
          println("indexedCacheRDD.count = " + indexedCacheRDD.count())*/

          /**
           * Step 2 - Map each partition to produce: idx -> (newPrimals, newCache, optionalModel)
           * Note that the optionalModel column is sparse. There exist only `numPartitions` of them in the RDD.
           */

          // if (indexedLocalProcessedData != null)
          // indexedLocalProcessedData.unpersist(false)

          indexedLocalProcessedData =
            indexedJointData.mapPartitionsWithIndex(
              (idx, dataIterator) =>
                mapper((idx, numPartitions),
                  dataIterator,
                  helperFunctions,
                  solverOptions,
                  globalModel,
                  globalModelWeightedAverage,
                  dataSize,
                  kAccum),
              preservesPartitioning = true)
              .cache()

          /**
           * Step 2.5 - A long lineage may cause a StackOverFlow error in the JVM.
           * So, trigger a checkpointing once in a while.
           */
          if (roundNum % solverOptions.checkpointFreq == 0) {
            indexedPrimalsRDD.checkpoint()
            indexedCacheRDD.checkpoint()
            indexedLocalProcessedData.checkpoint()
          }

          /**
           * Step 3a - Obtain the new global model
           * Collect models from all partitions and compute the new model locally on master
           */

          val localSummaryList =
            indexedLocalProcessedData
              .flatMapValues(_.localSummary)
              .values
              .collect()

          val sumDeltaWeightsAndEll =
            localSummaryList
              .map {
                case summary =>
                  val model = summary.deltaLocalModel
                  (model.getWeights(), model.getEll())
              }.reduce(
                (model1, model2) =>
                  (model1._1 + model2._1, model1._2 + model2._2))

          // Weighted Average model
          val sumDeltaWeightsAndEllWAvg =
            localSummaryList
              .map {
                case summary =>
                  val model = summary.deltaLocalAveragedModel
                  (model.getWeights(), model.getEll())
              }.reduce(
                (model1, model2) =>
                  (model1._1 + model2._1, model1._2 + model2._2))

          val deltaK: Vector[Int] = localSummaryList
            .map(_.deltaLocalK)
            .reduce((x, y) => x + y)
          kAccum += deltaK

          val newGlobalModel = globalModel.clone()
          newGlobalModel.updateWeights(globalModel.getWeights() + sumDeltaWeightsAndEll._1 * (beta / numPartitions))
          newGlobalModel.updateEll(globalModel.getEll() + sumDeltaWeightsAndEll._2 * (beta / numPartitions))

          // Weighted Average model
          val newGlobalModelWAvg = globalModelWeightedAverage.clone()
          newGlobalModelWAvg.updateWeights(globalModelWeightedAverage.getWeights() + sumDeltaWeightsAndEllWAvg._1 * (beta / numPartitions))
          newGlobalModelWAvg.updateEll(globalModelWeightedAverage.getEll() + sumDeltaWeightsAndEllWAvg._2 * (beta / numPartitions))

          val w_t = globalModel.getWeights()
          val w_tp1 = newGlobalModel.getWeights()

          // || w_t ||
          val w_t_norm = norm(w_t, 2)

          // || w_{t+1} - w_t} ||
          val w_diff_norm = norm(w_tp1 - w_t, 2)

          // cos( w_t , w_{t-1} )
          val cos_w = (w_t dot w_tp1) / (norm(w_t, 2) * norm(w_tp1, 2))

          val wDebugStr = "#wv %d,%f,%f,%f,%f\n".format(roundNum,
            getElapsedTimeSecs(),
            w_t_norm,
            w_diff_norm,
            cos_w)

          globalModel = newGlobalModel
          globalModelWeightedAverage = newGlobalModelWAvg

          /**
           * Step 3b - Obtain the new set of primals
           */

          val newPrimalsRDD = indexedLocalProcessedData
            .mapValues(_.primalInfo)

          indexedPrimalsRDD = indexedPrimalsRDD
            .leftOuterJoin(newPrimalsRDD)
            .mapValues {
              case ((prevW, prevEll), Some((newW, newEll))) =>
                (prevW + (newW * (beta / numPartitions)),
                  prevEll + (newEll * (beta / numPartitions)))
              case ((prevW, prevEll), None) => (prevW, prevEll)
            }.cache()

          /**
           * Step 3c - Obtain the new cache values
           */

          val newCacheRDD = indexedLocalProcessedData
            .mapValues(_.cache)

          indexedCacheRDD = indexedCacheRDD
            .leftOuterJoin(newCacheRDD)
            .mapValues {
              case (oldCache, Some(newCache)) => newCache.get
              case (oldCache, None)           => oldCache
            }.cache()

          /**
           * Debug info
           */
          // Obtain duality gap after each communication round
          val debugModel: StructSVMModel[X, Y] =
            if (solverOptions.doWeightedAveraging)
              globalModelWeightedAverage.clone()
            else globalModel.clone()

          // Is criteria for debugging met?
          val doDebugCalc: Boolean =
            if (solverOptions.debugMultiplier == 1) {
              true
            } else if (roundNum <= DEBUG_THRESH && roundNum == nextDebugRound) {
              nextDebugRound = nextDebugRound * solverOptions.debugMultiplier
              true
            } else if (roundNum > DEBUG_THRESH && roundNum % DEBUG_STEP == 0) {
              nextDebugRound += DEBUG_STEP
              true
            } else
              false

          val roundEvaluation =
            if (solverOptions.debug && doDebugCalc) {
              evaluateModel(debugModel, roundNum, w_t_norm, w_diff_norm, cos_w)
            } else {
              // If debug flag isn't on, perform calculations that don't trigger a shuffle
              val dual = -SolverUtils.objectiveFunction(debugModel.getWeights(), debugModel.getEll(), solverOptions.lambda)
              val elapsedTime = getElapsedTimeSecs()

              val wallTime = elapsedTime - (evaluateModelTimeMillis / 1000.0)
              val scoreFiller = Array.fill(dissolveFunctions.numClasses())(Double.NaN)
              RoundEvaluation(roundNum,
                elapsedTime, wallTime, Double.NaN, dual,
                Double.NaN, Double.NaN, Double.NaN, Double.NaN, Double.NaN,
                w_t_norm, w_diff_norm, cos_w, scoreFiller, 0.0, scoreFiller, 0.0)
            }

          debugSb ++= roundEvaluation + "\n"
      }

    (globalModel, debugSb.toString())
  }

  def mapper(partitionInfo: (Int, Int), // (partitionIdx, numPartitions)
             dataIterator: Iterator[(Index, InputDataShard[X, Y])],
             helperFunctions: HelperFunctions[X, Y],
             solverOptions: SolverOptions[X, Y],
             localModel: StructSVMModel[X, Y],
             localModelWeightedAverage: StructSVMModel[X, Y],
             n: Int,
             kAccum: Vector[Int]): Iterator[(Index, ProcessedDataShard[X, Y])] = {

    // println("[Round %d] Beginning mapper at partition %d".format(roundNum, partitionNum))
    case class UpdateQuantities(w_s: Vector[Double],
                                ell_s: Double,
                                gamma: Double)

    /**
     * Return updates for a specific choice of argmax and weights
     */
    def getUpdateQuantities(model: StructSVMModel[X, Y],
                            pattern: X,
                            label: Y,
                            ystar_i: Y,
                            w_i: Vector[Double],
                            ell_i: Double,
                            k: Int): UpdateQuantities = {
      val lambda = solverOptions.lambda
      val lossFn = helperFunctions.lossFn
      val phi = helperFunctions.featureFn

      val phi_i_label: Vector[Double] = time({ phi(pattern, label) }, "phi")
      val phi_i_ystar: Vector[Double] = phi(pattern, ystar_i)
      val psi_i: Vector[Double] = phi_i_label - phi_i_ystar
      val w_s: Vector[Double] = psi_i :* (1.0 / (n * lambda))
      val loss_i: Double = time({ lossFn(label, ystar_i) }, "Delta")
      val ell_s: Double = (1.0 / n) * loss_i

      val gamma: Double =
        if (solverOptions.doLineSearch) {
          val thisModel = model
          val gamma_opt = (thisModel.getWeights().t * (w_i - w_s) - ((ell_i - ell_s) * (1.0 / lambda))) /
            ((w_i - w_s).t * (w_i - w_s) + EPS)
          max(0.0, min(1.0, gamma_opt))
        } else {
          (2.0 * n) / (k + 2.0 * n)
        }

      UpdateQuantities(w_s, ell_s, gamma)
    }

    val maxOracle = helperFunctions.oracleFn
    val oracleStreamFn = helperFunctions.oracleStreamFn
    val phi = helperFunctions.featureFn
    val lossFn = helperFunctions.lossFn
    val fineOracleFn = helperFunctions.fineOracleFn

    val lambda = solverOptions.lambda

    val (partitionIdx, numPartitions) = partitionInfo
    var k = kAccum(partitionIdx)

    var ell = localModel.getEll()
    var ellWeightedAverage = localModelWeightedAverage.getEll()

    val prevModel = localModel.clone()
    val prevModelWeightedAverage = localModelWeightedAverage.clone()

    for ((index, shard) <- dataIterator) yield {

      /*if (index < 10)
        println("Partition = %d, Index = %d".format(partitionNum, index))*/

      // 1) Pick example
      val pattern: X = shard.labeledObject.pattern
      val label: Y = shard.labeledObject.label

      // shard.primalInfo: (w_i, ell_i)
      val w_i = shard.primalInfo._1
      val ell_i = shard.primalInfo._2

      // println("w_i is sparse - " + w_i.isInstanceOf[SparseVector[Double]])

      // 2.a) Search for candidates
      val optionalCache_i: Option[BoundedCacheList[Y]] = shard.cache
      val bestCachedCandidateForI: Option[Y] =
        if (solverOptions.enableOracleCache && optionalCache_i.isDefined) {
          val fixedGamma: Double = (2.0 * n) / (k + 2.0 * n)

          val candidates: Seq[(Double, Int)] =
            optionalCache_i.get
              .map(y_i => (((phi(pattern, label) - phi(pattern, y_i)) :* (1 / (n * lambda))),
                (1.0 / n) * lossFn(label, y_i))) // Map each cached y_i to their respective (w_s, ell_s)
              .map {
                case (w_s, ell_s) =>
                  (localModel.getWeights().t * (w_i - w_s) - ((ell_i - ell_s) * (1 / lambda))) /
                    ((w_i - w_s).t * (w_i - w_s) + EPS) // Map each (w_s, ell_s) to their respective step-size values 
              }
              .zipWithIndex // We'll need the index later to retrieve the respective approx. ystar_i
              .filter { case (gamma, idx) => gamma > 0.0 }
              .map { case (gamma, idx) => (min(1.0, gamma), idx) } // Clip to [0,1] interval
              .filter { case (gamma, idx) => gamma >= 0.5 * fixedGamma } // Further narrow down cache contenders
              .sortBy { case (gamma, idx) => gamma }

          // If there is a good contender among the cached datapoints, return it
          if (candidates.size >= 1)
            Some(optionalCache_i.get(candidates.head._2))
          else None
        } else None

      // 2.b) Solve loss-augmented inference for point i
      val yCacheMaxLevel =
        if (bestCachedCandidateForI.isEmpty) {

          val argmaxStream = oracleStreamFn(localModel, pattern, label)

          // val GAMMA_THRESHOLD = 0.5 * ((2.0 * n) / (k + 2.0 * n))
          val GAMMA_THRESHOLD = EPS
          // Sort by gamma, in decreasing order
          def diff(y: (Y, UpdateQuantities)): Double = -y._2.gamma
          // Maintain a priority queue, where the head contains the argmax with highest gamma value
          val argmaxCandidates = new PriorityQueue[(Y, UpdateQuantities)]()(Ordering.by(diff))

          // Request for argmax candidates, till a good candidate is found
          var startConsume = System.currentTimeMillis()
          var prevConsume = System.currentTimeMillis()

          time({
            argmaxStream
              .takeWhile {
                // Continue requesting for candidate argmax, till a good candidate is found (gamma > 0)
                case argmax_y =>
                  val updates = getUpdateQuantities(localModel, pattern, label, argmax_y, w_i, ell_i, k)
                  argmaxCandidates.enqueue((argmax_y, updates))
                  val consumeNext = updates.gamma <= GAMMA_THRESHOLD
                  consumeNext
              }.foreach { // Streams are lazy, force an `action` on them. Otherwise, subsequent elements
                // do not get computed
                identity // Do nothing, just consume the argmax and move on
              }
          }, "oracle-stream")

          val (ystar, updates): (Y, UpdateQuantities) =
            time({ argmaxCandidates.head }, "argmax-head")

          val maxLevel = argmaxCandidates.size - 1 // Levels are 0-indexed

          val updatedCache: Option[BoundedCacheList[Y]] =
            if (solverOptions.enableOracleCache) {

              val nonTruncatedCache =
                if (optionalCache_i.isDefined)
                  optionalCache_i.get :+ ystar
                else
                  MutableList[Y]() :+ ystar

              // Truncate cache to given size and pack it as an Option
              Some(nonTruncatedCache.takeRight(solverOptions.oracleCacheSize))
            } else None

          (ystar, updatedCache, maxLevel)
        } else {
          (bestCachedCandidateForI.get, optionalCache_i, 0)
        }

      val ystar_i = yCacheMaxLevel._1
      val updatedCache = yCacheMaxLevel._2
      val maxLevel = yCacheMaxLevel._3

      val updates = getUpdateQuantities(localModel, pattern, label, ystar_i, w_i, ell_i, k)
      val gamma = updates.gamma
      val w_s = updates.w_s
      val ell_s = updates.ell_s

      // Calculate Energy in this level
      val phi_i_label: Vector[Double] = phi(pattern, label)
      val phi_i_ystar: Vector[Double] = phi(pattern, ystar_i)
      val psi_i: Vector[Double] = phi_i_label - phi_i_ystar
      val energy = lossFn(label, ystar_i) - (localModel.getWeights() dot psi_i)

      val gammaLogSb = new StringBuilder()
      gammaLogSb ++=
        "[G] " + k + "," +
        System.currentTimeMillis() + "," +
        maxLevel + "," +
        helperFunctions.xid(pattern) + "," +
        gamma + "," +
        norm(w_s, 2) + "," +
        ell_s + "," +
        energy

      // Obtain oracle decoding for last-level, in order to compare the gamma
      val ystar_i_fine = fineOracleFn(localModel, pattern, label)
      if (ystar_i_fine != null) {
        val updates_fine = getUpdateQuantities(localModel, pattern, label, ystar_i_fine, w_i, ell_i, k)
        val gamma_fine = updates_fine.gamma
        val w_s_fine = updates.w_s
        val ell_s_fine = updates_fine.ell_s

        val phi_i_ystar_fine: Vector[Double] = phi(pattern, ystar_i_fine)
        val psi_i_fine: Vector[Double] = phi_i_label - phi_i_ystar_fine
        val energy_fine = lossFn(label, ystar_i_fine) - (localModel.getWeights() dot psi_i_fine)

        gammaLogSb ++= "," +
          gamma_fine + "," +
          norm(w_s_fine, 2) + "," +
          ell_s_fine + "," +
          energy_fine
      } else {
        gammaLogSb ++= "," +
          Double.NaN + "," +
          Double.NaN + "," +
          Double.NaN + "," +
          Double.NaN
      }

      LAdap.log.info(gammaLogSb.toString())

      val tempWeights1: Vector[Double] = localModel.getWeights() - w_i
      localModel.updateWeights(tempWeights1)
      val w_i_prime = w_i * (1.0 - gamma) + (w_s * gamma)
      val tempWeights2: Vector[Double] = localModel.getWeights() + w_i_prime
      localModel.updateWeights(tempWeights2)

      ell = ell - ell_i
      val ell_i_prime = (ell_i * (1.0 - gamma)) + (ell_s * gamma)
      ell = ell + ell_i_prime

      // Do Weighted Averaging
      val rho = 2.0 / (k + 2.0)
      val wAvg = (1.0 - rho) * localModelWeightedAverage.getWeights() + rho * localModel.getWeights()
      val ellAvg = (1.0 - rho) * localModelWeightedAverage.getEll() + rho * localModel.getEll()
      localModelWeightedAverage.updateWeights(wAvg)
      localModelWeightedAverage.updateEll(ellAvg)

      k += 1

      if (!dataIterator.hasNext) {

        localModel.updateEll(ell)

        val deltaLocalModel = localModel.clone()
        deltaLocalModel.updateWeights(localModel.getWeights() - prevModel.getWeights())
        deltaLocalModel.updateEll(localModel.getEll() - prevModel.getEll())

        val deltaLocalModelWeightedAverage = localModelWeightedAverage.clone()
        deltaLocalModelWeightedAverage.updateWeights(localModelWeightedAverage.getWeights() - prevModelWeightedAverage.getWeights())
        deltaLocalModelWeightedAverage.updateEll(localModelWeightedAverage.getEll() - prevModelWeightedAverage.getEll())

        val deltaK = k - kAccum(partitionIdx)
        val kAccumLocalDelta = DenseVector.zeros[Int](numPartitions)
        kAccumLocalDelta(partitionIdx) = deltaK

        (index, ProcessedDataShard((w_i_prime - w_i, ell_i_prime - ell_i), updatedCache, Some(LocalSummary(deltaLocalModel, kAccumLocalDelta, deltaLocalModelWeightedAverage))))
      } else
        (index, ProcessedDataShard((w_i_prime - w_i, ell_i_prime - ell_i), updatedCache, None))
    }
  }

}