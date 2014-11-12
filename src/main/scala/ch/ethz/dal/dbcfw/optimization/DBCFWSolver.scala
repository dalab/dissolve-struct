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

/**
 * LogHelper is a trait you can mix in to provide easy log4j logging
 * for your scala classes.
 */
/*trait LogHelper {
  val loggerName = this.getClass.getName
  lazy val logger = Logger.getLogger(loggerName)
}*/

class DBCFWSolver[X, Y](
  @transient val sc: SparkContext,
  val data: Vector[LabeledObject[X, Y]],
  val featureFn: (Y, X) => Vector[Double], // (y, x) => FeatureVect, 
  val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossVal, 
  val oracleFn: (StructSVMModel[X, Y], Y, X) => Y, // (model, y_i, x_i) => Lab, 
  val predictFn: (StructSVMModel[X, Y], X) => Y,
  val solverOptions: SolverOptions[X, Y],
  val miniBatchEnabled: Boolean) extends Serializable {

  /**
   * This runs on the Master node, and each round triggers a map-reduce job on the workers
   */
  def optimize()(implicit m: ClassTag[Y]): (StructSVMModel[X, Y], String) = {

    // autoconfigure parameters
    val NUM_DECODING_SAMPLES = 5
    val NUM_COMMN_SAMPLES = 5 // Time taken for a single round of communication

    val debugSb: StringBuilder = new StringBuilder()

    val d: Int = featureFn(data(0).label, data(0).pattern).size
    // Let the initial model contain zeros for all weights
    var globalModel: StructSVMModel[X, Y] = new StructSVMModel[X, Y](SparseVector.zeros(d), 0.0, SparseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    /**
     *  Create two RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     *  3. indexedCacheRDD (Index, BoundedCacheList)
     */
    val indexedTrainData: Array[(Index, LabeledObject[X, Y])] = (0 until data.size).toArray.zip(data.toArray)
    val indexedPrimals: Array[(Index, PrimalInfo)] = (0 until data.size).toArray.zip(
      Array.fill(data.size)((SparseVector.zeros[Double](d), 0.0)) // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
      )

    val indexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      if (solverOptions.enableManualPartitionSize)
        sc.parallelize(indexedTrainData, solverOptions.NUM_PART)
      else
        sc.parallelize(indexedTrainData)

    var indexedPrimalsRDD: RDD[(Index, PrimalInfo)] =
      if (solverOptions.enableManualPartitionSize)
        sc.parallelize(indexedPrimals, solverOptions.NUM_PART)
      else
        sc.parallelize(indexedPrimals)

    // For each Primal (i.e, Index), cache a list of Decodings (i.e, Vector[Double])
    // If cache is disabled, add an empty array. This immediately drops the joins later on and saves time in communicating an unnecessary RDD.
    val indexedCache: Array[(Index, BoundedCacheList[Y])] =
      if (solverOptions.enableOracleCache)
        (0 until data.size).toArray.zip(
          Array.fill(data.size)(MutableList[Y]()) // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
          )
      else
        Array[(Index, BoundedCacheList[Y])]()

    var indexedCacheRDD: RDD[(Index, BoundedCacheList[Y])] =
      if (solverOptions.enableManualPartitionSize)
        sc.parallelize(indexedCache, solverOptions.NUM_PART)
      else
        sc.parallelize(indexedCache)

    indexedPrimalsRDD.checkpoint()

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
        math.min(solverOptions.H / data.size, 1.0)
      else {
        println("[WARNING] %s is not a valid option. Reverting to sampling 50% of the dataset")
        0.5
      }
    }

    println("Beginning training of %d data points in %d passes with lambda=%f".format(data.size, solverOptions.numPasses, solverOptions.lambda))

    var avgDecodeTime: Double = 0.0
    var avgCommunicationTime: Double = 0.0
    /**
     * Monitoring round
     * In this mode, the optimal H and numPasses is calculated based on:
     * a. Time taken for decoding
     * b. Time taken for a single round of communication
     */
    if (solverOptions.autoconfigure) {

      /**
       * Obtain average time required to decode
       */
      var decodeTimings =
        for (i <- 0 until NUM_DECODING_SAMPLES) yield {
          var randModel: StructSVMModel[X, Y] = new StructSVMModel[X, Y](DenseVector.rand(d), Math.random(), DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)
          var sampled_i = util.Random.nextInt(data.size)

          val startDecodingTime = System.currentTimeMillis()
          oracleFn(randModel, data(sampled_i).label, data(sampled_i).pattern)
          val endDecodingTime = System.currentTimeMillis()

          endDecodingTime - startDecodingTime
        }
      avgDecodeTime = decodeTimings.reduce((t1, t2) => t1 + t2).toDouble / NUM_DECODING_SAMPLES

      /**
       * Obtain average time required to finish 1 round of communication
       *
       * Run a dummy map-reduce job to get the timings
       */
      def dummyOracle(model: StructSVMModel[X, Y], yi: Y, xi: X): Y = yi

      val communicationTimings =
        for (i <- 0 until NUM_COMMN_SAMPLES) yield {

          // Run Mapper
          val temp: RDD[(StructSVMModel[X, Y], Array[(Index, (PrimalInfo, Option[BoundedCacheList[Y]]))], StructSVMModel[X, Y])] = indexedTrainDataRDD.sample(solverOptions.sampleWithReplacement, sampleFrac, solverOptions.randSeed)
            .join(indexedPrimalsRDD)
            .leftOuterJoin(indexedCacheRDD)
            .mapValues {
              case ((labeledObject, primalInfo), boundedCacheList) => (labeledObject, primalInfo, boundedCacheList) // Flatten values for readability
            }
            .mapPartitions(x => mapper(x, globalModel, featureFn, lossFn, dummyOracle,
              predictFn, solverOptions, miniBatchEnabled), preservesPartitioning = true)

          // Finish Reducer. Thus, finish one round of communication
          val startCommunicationTime = System.currentTimeMillis()
          val reducedData: (StructSVMModel[X, Y], RDD[(Index, PrimalInfo)], RDD[(Index, BoundedCacheList[Y])]) = reducer(temp, indexedPrimalsRDD, indexedCacheRDD, globalModel, d, beta = 1.0)
          val endCommunicationTime = System.currentTimeMillis()

          endCommunicationTime - startCommunicationTime
        }
      avgCommunicationTime = communicationTimings.reduce((t1, t2) => t1 + t2).toDouble / NUM_COMMN_SAMPLES

      /**
       * Given the average time to decode samples and communicate, figure out values of H and numPasses
       */
      /*println("Average decoding time: %f".format(avgDecodeTime))
      println("Average communication time: %f".format(avgCommunicationTime))*/

    }

    // logger.info("[DATA] round,time,train_error,test_error")
    val startTime = System.currentTimeMillis()
    debugSb ++= "round,time,primal,dual,gap,train_error,test_error\n"

    for (roundNum <- 1 to solverOptions.numPasses) {
      /**
       * Mapper
       */
      val temp: RDD[(StructSVMModel[X, Y], Array[(Index, (PrimalInfo, Option[BoundedCacheList[Y]]))], StructSVMModel[X, Y])] = indexedTrainDataRDD.sample(solverOptions.sampleWithReplacement, sampleFrac, solverOptions.randSeed)
        .join(indexedPrimalsRDD)
        .leftOuterJoin(indexedCacheRDD)
        .mapValues {
          case ((labeledObject, primalInfo), boundedCacheList) => (labeledObject, primalInfo, boundedCacheList) // Flatten values for readability
        }
        .mapPartitions(x => mapper(x, globalModel, featureFn, lossFn, oracleFn,
          predictFn, solverOptions, miniBatchEnabled), preservesPartitioning = true)

      /**
       * Reducer
       */
      val reducedData: (StructSVMModel[X, Y], RDD[(Index, PrimalInfo)], RDD[(Index, BoundedCacheList[Y])]) = reducer(temp, indexedPrimalsRDD, indexedCacheRDD, globalModel, d, beta = 1.0)

      // Update the global model and the primal for each i
      globalModel = reducedData._1
      indexedPrimalsRDD = reducedData._2
      indexedCacheRDD = reducedData._3

      val elapsedTime = (System.currentTimeMillis() - startTime).toDouble / 1000.0

      val trainError = SolverUtils.averageLoss(data, lossFn, predictFn, globalModel)
      val testError = SolverUtils.averageLoss(solverOptions.testData, lossFn, predictFn, globalModel)

      // Obtain duality gap after each communication round
      val debugModel: StructSVMModel[X, Y] = globalModel.clone()
      val f = -SolverUtils.objectiveFunction(debugModel.getWeights(), debugModel.getEll(), solverOptions.lambda)
      val gapTup = SolverUtils.dualityGap(data, featureFn, lossFn, oracleFn, debugModel, solverOptions.lambda)
      val gap = gapTup._1
      val primal = f + gap

      // logger.info("[DATA] %d,%f,%f,%f\n".format(roundNum, elapsedTime, trainError, testError))
      println("[Round #%d] Train loss = %f, Test loss = %f, Primal = %f, Gap = %f\n".format(roundNum, trainError, testError, primal, gap))
      val curTime = (System.currentTimeMillis() - startTime) / 1000
      debugSb ++= "%d,%d,%f,%f,%f,%f,%f\n".format(roundNum, curTime, primal, f, gap, trainError, testError)
    }

    println("Average decoding time: %f".format(avgDecodeTime))
    println("Average communication time: %f".format(avgCommunicationTime))

    println("globalModel.weights is sparse - " + globalModel.getWeights.isInstanceOf[SparseVector[Double]])
    println("w_i is sparse - " + indexedPrimalsRDD.first._2._1.isInstanceOf[SparseVector[Double]])

    (globalModel, debugSb.toString())
  }

  /**
   * Takes as input a set of data and builds a SSVM model trained using BCFW
   */
  def mapper(dataIterator: Iterator[(Index, (LabeledObject[X, Y], PrimalInfo, Option[BoundedCacheList[Y]]))],
    localModel: StructSVMModel[X, Y],
    featureFn: (Y, X) => Vector[Double], // (y, x) => FeatureVect, 
    lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossVal, 
    oracleFn: (StructSVMModel[X, Y], Y, X) => Y, // (model, y_i, x_i) => Lab, 
    predictFn: (StructSVMModel[X, Y], X) => Y,
    solverOptions: SolverOptions[X, Y],
    miniBatchEnabled: Boolean): Iterator[(StructSVMModel[X, Y], Array[(Index, (PrimalInfo, Option[BoundedCacheList[Y]]))], StructSVMModel[X, Y])] = {

    val prevModel: StructSVMModel[X, Y] = localModel.clone()

    val numPasses = solverOptions.numPasses
    val lambda = solverOptions.lambda
    val debugOn: Boolean = solverOptions.debug
    val xldebug: Boolean = solverOptions.xldebug

    /**
     * Reorganize data for training
     */
    val zippedData: Array[(Index, (LabeledObject[X, Y], PrimalInfo, Option[BoundedCacheList[Y]]))] = dataIterator.toArray.sortBy(_._1)
    val data: Array[LabeledObject[X, Y]] = zippedData.map(x => x._2._1)
    val globalDataIdx: Array[Index] = zippedData.map(x => x._1)
    // Mapping of indexMapping(localIndex) -> globalIndex
    val localToGlobal: Array[Index] = zippedData.map(x => x._1)

    // Alternate implementation - Use mutable maps. Immutable causes stack overflow
    val globalToLocal: mutable.Map[Index, Index] = mutable.Map.empty[Index, Index]
    for (ele <- localToGlobal.zipWithIndex)
      globalToLocal(ele._1) = ele._2

    // Create an Oracle Map: Index => BoundedCacheList
    val oracleCache = collection.mutable.Map[Int, BoundedCacheList[Y]]()
    if (solverOptions.enableOracleCache)
      zippedData.map {
        case (index, (labeledObject, primalInfo, boundedCacheList)) => oracleCache(globalToLocal(index)) = boundedCacheList.get
      }

    val maxOracle = oracleFn
    val phi = featureFn
    // Number of dimensions of \phi(x, y)
    val d: Int = localModel.getWeights().size

    // Only to keep track of the \Delta localModel
    val deltaLocalModel = new StructSVMModel(SparseVector.zeros(d), 0.0, SparseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    val eps: Double = 2.2204E-16

    var k: Int = 0
    val n: Int = data.size

    val wMat: Matrix[Double] = CSCMatrix.zeros[Double](d, n)
    val ellMat: Vector[Double] = SparseVector.zeros[Double](n)

    // Copy w_i's and l_i's into local wMat and ellMat
    for (i <- 0 until n) {
      updateColumn(wMat, zippedData(i)._2._2._1, i)
      ellMat(i) = zippedData(i)._2._2._2
    }
    val prevWMat: Matrix[Double] = wMat.copy
    val prevEllMat: Vector[Double] = ellMat.copy

    var ell: Double = localModel.getEll()
    localModel.updateEll(0.0)

    // Initialization in case of Weighted Averaging
    var wAvg: Vector[Double] =
      if (solverOptions.doWeightedAveraging)
        SparseVector.zeros(d)
      else null
    var lAvg: Double = 0.0

    for ((datapoint, globalIdx) <- data.zip(globalDataIdx)) {

      // Convert globalIdx to localIdx a.k.a "i"
      val i: Index = globalToLocal(globalIdx)

      // 1) Pick example
      val pattern: X = datapoint.pattern
      val label: Y = datapoint.label

      // 2.a) Search for candidates
      val bestCachedCandidateForI: Option[Y] =
        if (solverOptions.enableOracleCache && oracleCache.contains(i)) {
          val candidates: Seq[(Double, Int)] =
            oracleCache(i)
              .map(y_i => (((phi(label, pattern) - phi(y_i, pattern)) :* (1 / (n * lambda))),
                (1.0 / n) * lossFn(label, y_i))) // Map each cached y_i to their respective (w_s, ell_s)
              .map {
                case (w_s, ell_s) =>
                  val wcoli = getMatrixColumn(wMat, i)
                  (localModel.getWeights().t * (wcoli - w_s) - ((ellMat(i) - ell_s) * (1 / lambda))) /
                    ((wcoli - w_s).t * (wcoli - w_s) + eps) // Map each (w_s, ell_s) to their respective step-size values	
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

      // 2) Solve loss-augmented inference for point i
      val ystar_i: Y =
        if (bestCachedCandidateForI.isEmpty) {
          val ystar = maxOracle(localModel, label, pattern)

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

      // 3) Define the update quantities
      val psi_i: Vector[Double] = phi(label, pattern) - phi(ystar_i, pattern)
      val w_s: Vector[Double] = psi_i :* (1.0 / (n * lambda))
      val loss_i: Double = lossFn(label, ystar_i)
      val ell_s: Double = (1.0 / n) * loss_i

      // 4) Get step-size gamma
      val wcoli = getMatrixColumn(wMat, i)
      val gamma: Double =
        if (solverOptions.doLineSearch) {
          val thisModel = if (miniBatchEnabled) prevModel else localModel
          val gamma_opt = (thisModel.getWeights().t * (wcoli - w_s) - ((ellMat(i) - ell_s) * (1.0 / lambda))) /
            ((wcoli - w_s).t * (wcoli - w_s) + eps)
          max(0.0, min(1.0, gamma_opt))
        } else {
          (2.0 * n) / (k + 2.0 * n)
        }

      // 5, 6, 7, 8) Update the weights of the model
      if (miniBatchEnabled) {
        val wcoli = getMatrixColumn(wMat, i)
        updateColumn(wMat, wcoli * (1.0 - gamma) + (w_s * gamma), i)
        // wMat(::, i) := wMat(::, i) * (1.0 - gamma) + (w_s * gamma)
        ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma)
        deltaLocalModel.updateWeights(localModel.getWeights() + (getMatrixColumn(wMat, i) - getMatrixColumn(prevWMat, i)))
        deltaLocalModel.updateEll(localModel.getEll() + (ellMat(i) - prevEllMat(i)))
      } else {
        // In case of CoCoA
        val tempWeights1: Vector[Double] = localModel.getWeights() - getMatrixColumn(wMat, i)
        localModel.updateWeights(tempWeights1)
        deltaLocalModel.updateWeights(tempWeights1)
        val wcoli = getMatrixColumn(wMat, i)
        updateColumn(wMat, wcoli * (1.0 - gamma) + (w_s * gamma), i)
        val tempWeights2: Vector[Double] = localModel.getWeights() + getMatrixColumn(wMat, i)
        localModel.updateWeights(tempWeights2)
        deltaLocalModel.updateWeights(tempWeights2)

        ell = ell - ellMat(i)
        ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma)
        ell = ell + ellMat(i)
      }

      // 9) Optionally update the weighted average
      if (solverOptions.doWeightedAveraging) {
        val rho: Double = 2.0 / (k + 2.0)
        wAvg = (wAvg * (1.0 - rho)) + (localModel.getWeights * rho)
        lAvg = (lAvg * (1.0 - rho)) + (ell * rho)
      }

      k = k + 1

    }

    if (solverOptions.doWeightedAveraging) {
      localModel.updateWeights(wAvg)
      localModel.updateEll(lAvg)
    } else {
      localModel.updateEll(ell)
    }

    val localIndexedDeltaPrimals: Array[(Index, (PrimalInfo, Option[BoundedCacheList[Y]]))] = zippedData.map(_._1).map(k =>
      (k, // Index
        (((getMatrixColumn(wMat, globalToLocal(k)) - getMatrixColumn(prevWMat, globalToLocal(k))), // PrimalInfo.w
          ellMat(globalToLocal(k)) - prevEllMat(globalToLocal(k))), // PrimalInfo.ell
          oracleCache.get(globalToLocal(k))))) // Cache

    // If this flag is set, return only the change in w's
    localModel.updateWeights(localModel.getWeights() - prevModel.getWeights())
    localModel.updateEll(localModel.getEll() - prevModel.getEll())

    // Finally return a single element iterator
    { List.empty[(StructSVMModel[X, Y], Array[(Index, (PrimalInfo, Option[BoundedCacheList[Y]]))], StructSVMModel[X, Y])] :+ (localModel, localIndexedDeltaPrimals, deltaLocalModel) }.iterator
  }

  /**
   * Takes as input a number of SVM Models, along with Primal information for each data point, and combines them into a single Model and Primal block
   */
  def reducer( // sc: SparkContext,
    zippedModels: RDD[(StructSVMModel[X, Y], Array[(Index, (PrimalInfo, Option[BoundedCacheList[Y]]))], StructSVMModel[X, Y])], // The optimize step returns k blocks. Each block contains (\Delta LocalModel, [\Delta PrimalInfo_i]).
    oldPrimalInfo: RDD[(Index, PrimalInfo)],
    oldCache: RDD[(Index, BoundedCacheList[Y])],
    oldGlobalModel: StructSVMModel[X, Y],
    d: Int,
    beta: Double): (StructSVMModel[X, Y], RDD[(Index, PrimalInfo)], RDD[(Index, BoundedCacheList[Y])]) = {

    val k: Double = zippedModels.count.toDouble // This refers to the number of localModels generated

    // Here, map is applied k(=#workers) times
    val sumDeltaWeights =
      zippedModels.map(model => model._1.getWeights()).reduce((deltaWeightA, deltaWeightB) => deltaWeightA + deltaWeightB)
    val sumDeltaElls =
      zippedModels.map(model => model._1.getEll).reduce((ellA, ellB) => ellA + ellB)

    /**
     * Create the new global model
     */
    val newGlobalModel = new StructSVMModel(oldGlobalModel.getWeights() + (sumDeltaWeights / k) * beta,
      oldGlobalModel.getEll() + (sumDeltaElls / k) * beta,
      SparseVector.zeros(d),
      oldGlobalModel.featureFn,
      oldGlobalModel.lossFn,
      oldGlobalModel.oracleFn,
      oldGlobalModel.predictFn)

    /**
     * Merge all the w_i's and l_i's
     *
     * First flatMap returns a [newDeltaPrimalInfo_k]. This is applied k times, returns a sequence of n deltaPrimalInfos
     *
     * By doing a right outer join, we ensure that all the indices are retained, even in case data points are sampled
     *
     * After join, we have a sequence of (Index, (PrimalInfo_A, PrimalInfo_B))
     * where PrimalInfo_A = PrimalInfo_i at t-1
     * and   PrimalInfo_B = \Delta PrimalInfo_i
     */
    val primalsAndCache = zippedModels.flatMap {
      case (model, primalsAndCache, debugModel) =>
        primalsAndCache
    }

    val indexedPrimals: RDD[(Index, PrimalInfo)] =
      primalsAndCache.mapValues { case (primals, cache) => primals }
        .rightOuterJoin(oldPrimalInfo)
        .mapValues {
          case (Some((newW, newEll)), (prevW, prevEll)) =>
            (prevW + (newW * (beta / k)),
              prevEll + (newEll * (beta / k)))
          case (None, (prevW, prevEll)) => (prevW, prevEll)
        }

    val indexedCache =
      if (solverOptions.enableOracleCache)
        primalsAndCache.mapValues { case (primals, cache) => cache }
          .rightOuterJoin(oldCache)
          .mapValues {
            // newCache includes entries in the previous cache too
            case (Some(newCache), oldCache) => newCache.get
            case (None, oldCache) => oldCache
          }
      else
        oldCache

    // indexedPrimals isn't materialized till an RDD action is called. Force this by calling one.
    indexedPrimals.checkpoint()
    indexedPrimals.count()
    indexedCache.count()

    (newGlobalModel, indexedPrimals, indexedCache)
  }

}