package ch.ethz.dal.dbcfw.optimization

import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import org.apache.spark.SparkContext
import ch.ethz.dal.dbcfw.classification.Types._
import org.apache.spark.SparkContext._
import org.apache.log4j.Logger
import scala.collection.mutable
import scala.collection.mutable.MutableList
import ch.ethz.dal.dbcfw.utils.LinAlgOps._
import scala.reflect.ClassTag
import org.apache.spark.HashPartitioner

class DBCFWSolverTuned[X, Y](
  val data: RDD[LabeledObject[X, Y]],
  val featureFn: (Y, X) => Vector[Double], // (y, x) => FeatureVect, 
  val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossVal, 
  val oracleFn: (StructSVMModel[X, Y], Y, X) => Y, // (model, y_i, x_i) => Lab, 
  val predictFn: (StructSVMModel[X, Y], X) => Y,
  val solverOptions: SolverOptions[X, Y],
  val miniBatchEnabled: Boolean) extends Serializable {

  /**
   * Some case classes to make code more readable
   */

  case class HelperFunctions[X, Y](featureFn: (Y, X) => Vector[Double],
                                   lossFn: (Y, Y) => Double,
                                   oracleFn: (StructSVMModel[X, Y], Y, X) => Y,
                                   predictFn: (StructSVMModel[X, Y], X) => Y)

  // Input to the mapper: idx -> DataShard
  case class InputDataShard[X, Y](labeledObject: LabeledObject[X, Y],
                                  primalInfo: PrimalInfo,
                                  cache: Option[BoundedCacheList[Y]])

  // Output of the mapper: idx -> ProcessedDataShard
  case class ProcessedDataShard[X, Y](primalInfo: PrimalInfo,
                                      cache: Option[BoundedCacheList[Y]],
                                      deltaLocalModel: Option[StructSVMModel[X, Y]])

  /**
   * This runs on the Master node, and each round triggers a map-reduce job on the workers
   */
  def optimize()(implicit m: ClassTag[Y]): (StructSVMModel[X, Y], String) = {

    val sc = data.context

    val debugSb: StringBuilder = new StringBuilder()

    val samplePoint = data.first()
    val dataSize = data.count().toInt

    val verboseDebug: Boolean = false

    val d: Int = featureFn(samplePoint.label, samplePoint.pattern).size
    // Let the initial model contain zeros for all weights
    // Global model uses Dense Vectors by default
    var globalModel: StructSVMModel[X, Y] = new StructSVMModel[X, Y](DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn, solverOptions.numClasses)

    val numPartitions: Int =
      data.partitions.size

    val beta: Double = 1.0

    val helperFunctions: HelperFunctions[X, Y] = HelperFunctions(featureFn, lossFn, oracleFn, predictFn)

    /**
     *  Create three RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     *  3. indexedCacheRDD (Index, BoundedCacheList)
     *  4. indexedLocalProcessedData (Index, LocallyProcessedData)
     */
    val indexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      data.zipWithIndex()
        .map {
          case (labeledObject, idx) =>
            (idx.toInt, labeledObject)
        }
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

    debugSb ++= "# indexedTrainDataRDD.partitions.size=%d\n".format(indexedTrainDataRDD.partitions.size)
    debugSb ++= "# indexedPrimalsRDD.partitions.size=%d\n".format(indexedPrimalsRDD.partitions.size)

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
        println("[WARNING] %s is not a valid option. Reverting to sampling 50% of the dataset")
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

    var weightedAveragesOfPrimals: PrimalInfo =
      if (solverOptions.doWeightedAveraging)
        (DenseVector.zeros(d), 0.0)
      else null

    var iterCount: Int = 0

    println("Beginning training of %d data points in %d passes with lambda=%f".format(dataSize, solverOptions.numPasses, solverOptions.lambda))

    val startTime = System.currentTimeMillis()
    debugSb ++= "round,time,primal,dual,gap,train_error,test_error\n"

    /**
     * ==== Begin Training rounds ====
     */
    for (roundNum <- 1 to solverOptions.numPasses) {

      /**
       * Step 1 - Create a joint RDD containing all information of idx -> (data, primals, cache)
       */

      // TODO Any performance benefits of using Int instead of Index, and 3-Tuple instead of DataShard?
      val indexedJointData: RDD[(Index, InputDataShard[X, Y])] =
        indexedTrainDataRDD
          .join(indexedPrimalsRDD)
          .leftOuterJoin(indexedCacheRDD)
          .mapValues { // Because mapValues preserves partitioning
            case ((labeledObject, primalInfo), cache) =>
              InputDataShard(labeledObject, primalInfo, cache)
          }

      /**
       * Step 2 - Map each partition to produce: idx -> (newPrimals, newCache, optionalModel)
       * Note that the optionalModel column is sparse. There exist only `numPartitions` of them in the RDD.
       */

      // if (indexedLocalProcessedData != null)
      // indexedLocalProcessedData.unpersist(false)

      indexedLocalProcessedData =
        indexedJointData.mapPartitionsWithIndex(
          (idx, dataIterator) =>
            mapper(idx,
              dataIterator,
              helperFunctions,
              solverOptions,
              globalModel,
              dataSize,
              roundNum),
          preservesPartitioning = true)
          .cache()

      /**
       * Step 3a - Obtain the new global model
       * Collect models from all partitions and compute the new model locally on master
       */

      val newGlobalModelList =
        indexedLocalProcessedData
          .filter {
            case (idx, shard) => shard.deltaLocalModel.isDefined
          }
          .mapValues(_.deltaLocalModel.get)
          .values
          .collect()

      val sumDeltaWeightsAndEll =
        newGlobalModelList
          .map {
            case model =>
              (model.getWeights(), model.getEll())
          }.reduce(
            (model1, model2) =>
              (model1._1 + model2._1, model1._2 + model2._2))

      val newGlobalModel = globalModel.clone()
      newGlobalModel.updateWeights(globalModel.getWeights() + sumDeltaWeightsAndEll._1 * (beta / numPartitions))
      newGlobalModel.updateEll(globalModel.getEll() + sumDeltaWeightsAndEll._2 * (beta / numPartitions))
      globalModel = newGlobalModel

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

      /*
      println("Size of indexedTrainDataRDD = " + indexedTrainDataRDD.count())
      println("Size of indexedJointData = " + indexedJointData.count())
      println("Size of indexedLocalProcessedData = " + indexedLocalProcessedData.count())
      println("Size of indexedPrimalsRDD = " + indexedPrimalsRDD.count())
      println("Size of indexedCacheRDD = " + indexedCacheRDD.count())
      */

      /**
       * Debug info
       */
      val elapsedTime = (System.currentTimeMillis() - startTime).toDouble / 1000.0

      // Obtain duality gap after each communication round
      val debugModel: StructSVMModel[X, Y] = globalModel.clone()
      if (solverOptions.doWeightedAveraging) {
        debugModel.updateWeights(weightedAveragesOfPrimals._1)
        debugModel.updateEll(weightedAveragesOfPrimals._2)
      }

      if (verboseDebug) {
        println("Model weights: " + debugModel.getWeights()(0 to 5).toDenseVector)
        debugSb ++= "Model weights: " + debugModel.getWeights()(0 to 5).toDenseVector + "\n"
      }

      val trainError = SolverUtils.averageLoss(data, lossFn, predictFn, debugModel)
      val testError =
        if (solverOptions.testDataRDD.isDefined)
          SolverUtils.averageLoss(solverOptions.testDataRDD.get, lossFn, predictFn, debugModel)
        else
          0.00

      val f = -SolverUtils.objectiveFunction(debugModel.getWeights(), debugModel.getEll(), solverOptions.lambda)
      val gapTup = SolverUtils.dualityGap(data, featureFn, lossFn, oracleFn, debugModel, solverOptions.lambda)
      val gap = gapTup._1
      val primal = f + gap

      // assert(gap >= 0.0, "Gap is negative")

      if (verboseDebug)
        debugSb ++= "# sum(w): %f, ell: %f\n".format(debugModel.getWeights().sum, debugModel.getEll())

      // logger.info("[DATA] %d,%f,%f,%f\n".format(roundNum, elapsedTime, trainError, testError))
      println("[Round #%d] Train loss = %f, Test loss = %f, Primal = %f, Gap = %f\n".format(roundNum, trainError, testError, primal, gap))
      val curTime = (System.currentTimeMillis() - startTime) / 1000
      debugSb ++= "%d,%d,%f,%f,%f,%f,%f\n".format(roundNum, curTime, primal, f, gap, trainError, testError)

      println("-----------------------------------------")

    }

    (globalModel, debugSb.toString())
  }

  def mapper2(partitionNum: Int,
              dataIterator: Iterator[(Index, InputDataShard[X, Y])],
              helperFunctions: HelperFunctions[X, Y],
              solverOptions: SolverOptions[X, Y],
              localModel: StructSVMModel[X, Y],
              n: Int,
              roundNum: Int): Iterator[(Index, ProcessedDataShard[X, Y])] = {

    println("[Round %d] Beginning mapper at partition %d".format(roundNum, partitionNum))

    val foo = dataIterator.next()
    val idx = foo._1
    val newprimals = foo._2.primalInfo
    val cache = foo._2.cache

    // (index, ProcessedDataShard((w_i, ell_i), shard.cache, Some(localModel)))
    { List.empty[(Index, ProcessedDataShard[X, Y])] :+ (idx, ProcessedDataShard(newprimals, cache, Some(localModel))) }.toIterator
  }

  def mapper(partitionNum: Int,
             dataIterator: Iterator[(Index, InputDataShard[X, Y])],
             helperFunctions: HelperFunctions[X, Y],
             solverOptions: SolverOptions[X, Y],
             localModel: StructSVMModel[X, Y],
             n: Int,
             roundNum: Int): Iterator[(Index, ProcessedDataShard[X, Y])] = {

    // println("[Round %d] Beginning mapper at partition %d".format(roundNum, partitionNum))

    val eps: Double = 2.2204E-16

    val maxOracle = helperFunctions.oracleFn
    val phi = helperFunctions.featureFn

    val lambda = solverOptions.lambda

    var k = 0
    var ell = localModel.getEll()

    for ((index, shard) <- dataIterator) yield {

      val pattern: X = shard.labeledObject.pattern
      val label: Y = shard.labeledObject.label

      // shard.primalInfo: (w_i, ell_i)
      val w_i = shard.primalInfo._1
      val ell_i = shard.primalInfo._2

      val ystar_i: Y = maxOracle(localModel, label, pattern)

      // 3) Define the update quantities
      val psi_i: Vector[Double] = phi(label, pattern) - phi(ystar_i, pattern)
      val w_s: Vector[Double] = psi_i :* (1.0 / (n * lambda))
      val loss_i: Double = lossFn(label, ystar_i)
      val ell_s: Double = (1.0 / n) * loss_i

      // 4) Get step-size gamma
      val gamma: Double =
        if (solverOptions.doLineSearch) {
          val thisModel = localModel
          val gamma_opt = (thisModel.getWeights().t * (w_i - w_s) - ((ell_i - ell_s) * (1.0 / lambda))) /
            ((w_i - w_s).t * (w_i - w_s) + eps)
          max(0.0, min(1.0, gamma_opt))
        } else {
          (2.0 * n) / (k + 2.0 * n)
        }

      val tempWeights1: Vector[Double] = localModel.getWeights() - w_i
      localModel.updateWeights(tempWeights1)
      val w_i_prime = w_i * (1.0 - gamma) + (w_s * gamma)
      val tempWeights2: Vector[Double] = localModel.getWeights() + w_i_prime
      localModel.updateWeights(tempWeights2)

      ell = ell - ell_i
      val ell_i_prime = (ell_i * (1.0 - gamma)) + (ell_s * gamma)
      ell = ell + ell_i_prime

      k += 1

      if (!dataIterator.hasNext) {
        /*println("Partition = " + partitionNum)
        println("k = " + k)
        println("w = " + localModel.getWeights()(0 until 5).toDenseVector)
        println("ell = " + ell)
        println()*/
        localModel.updateEll(ell)
        (index, ProcessedDataShard((w_i_prime - w_i, ell_i_prime - ell_i), shard.cache, Some(localModel)))
      } else
        (index, ProcessedDataShard((w_i_prime - w_i, ell_i_prime - ell_i), shard.cache, None))
    }
  }

}