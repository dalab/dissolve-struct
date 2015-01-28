package ch.ethz.dalab.dissolve.optimization

import org.apache.spark.rdd.RDD

import breeze.linalg._
import breeze.numerics._
import ch.ethz.dalab.dissolve.optimization.SolverUtils;
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.classification.StructSVMModel

import org.apache.spark.SparkContext

import ch.ethz.dalab.dissolve.classification.Types._

import org.apache.spark.SparkContext._
import org.apache.log4j.Logger

import scala.collection.mutable
import scala.collection.mutable.MutableList
import ch.ethz.dalab.dissolve.utils.LinAlgOps._
import scala.reflect.ClassTag

class DBCFWSolver[X, Y](
  val data: RDD[LabeledObject[X, Y]],
  val featureFn: (X, Y) => Vector[Double], // (y, x) => FeatureVect, 
  val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossVal, 
  val oracleFn: (StructSVMModel[X, Y], X, Y) => Y, // (model, y_i, x_i) => Lab, 
  val predictFn: (StructSVMModel[X, Y], X) => Y,
  val solverOptions: SolverOptions[X, Y],
  val miniBatchEnabled: Boolean) extends Serializable {

  /**
   * Some case classes to make code more readable
   */
  case class DataShard[X, Y](data: LabeledObject[X, Y], primalInfo: PrimalInfo, cache: Option[BoundedCacheList[Y]])
  case class LocalIndexedPrimalsCache[Y](primalInfo: PrimalInfo, cache: Option[BoundedCacheList[Y]])
  case class LocalProcessedShard[X, Y](localModel: StructSVMModel[X, Y],
                                       localIndexedDeltaPrimals: Array[(Index, LocalIndexedPrimalsCache[Y])],
                                       deltaLocalModel: StructSVMModel[X, Y],
                                       localWeightedAAvgOfPrimals: PrimalInfo,
                                       k: Int)

  /**
   * This runs on the Master node, and each round triggers a map-reduce job on the workers
   */
  def optimize()(implicit m: ClassTag[Y]): (StructSVMModel[X, Y], String) = {

    val sc = data.context

    val debugSb: StringBuilder = new StringBuilder()

    val samplePoint = data.first()
    val dataSize = data.count().toInt

    val verboseDebug: Boolean = false

    val d: Int = featureFn(samplePoint.pattern, samplePoint.label).size
    // Let the initial model contain zeros for all weights
    // Global model uses Dense Vectors by default
    var globalModel: StructSVMModel[X, Y] = new StructSVMModel[X, Y](DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn, solverOptions.numClasses)

    val numPartitions: Int =
      data.partitions.size

    /**
     *  Create three RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     *  3. indexedCacheRDD (Index, BoundedCacheList)
     */
    val indexedTrainDataRDD: RDD[(Index, LabeledObject[X, Y])] =
      data.zipWithIndex().map {
        case (labeledObject, idx) =>
          (idx.toInt, labeledObject)
      }

    val indexedPrimals: Array[(Index, PrimalInfo)] = (0 until dataSize).toArray.zip(
      // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
      Array.fill(dataSize)((
        if (solverOptions.sparse) // w_i can be either Sparse or Dense 
          SparseVector.zeros[Double](d)
        else
          DenseVector.zeros[Double](d),
        0.0)))
    var indexedPrimalsRDD: RDD[(Index, PrimalInfo)] = sc.parallelize(indexedPrimals, numPartitions)
    indexedPrimalsRDD.checkpoint()

    // For each Primal (i.e, Index), cache a list of Decodings (i.e, Y's)
    // If cache is disabled, add an empty array. This immediately drops the joins later on and saves time in communicating an unnecessary RDD.
    val indexedCache: Array[(Index, BoundedCacheList[Y])] =
      if (solverOptions.enableOracleCache)
        (0 until dataSize).toArray.zip(
          Array.fill(dataSize)(MutableList[Y]()) // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
          )
      else
        Array[(Index, BoundedCacheList[Y])]()
    var indexedCacheRDD: RDD[(Index, BoundedCacheList[Y])] = sc.parallelize(indexedCache, numPartitions)

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
       * Mapper
       */
      val temp: RDD[LocalProcessedShard[X, Y]] = indexedTrainDataRDD.sample(solverOptions.sampleWithReplacement, sampleFrac, solverOptions.randSeed)
        .join(indexedPrimalsRDD)
        .leftOuterJoin(indexedCacheRDD)
        .mapValues {
          case ((labeledObject, primalInfo), boundedCacheList) => DataShard(labeledObject, primalInfo, boundedCacheList) // Flatten values for readability
        }
        .mapPartitions(x => mapper(x, globalModel, featureFn, lossFn, oracleFn,
          predictFn, solverOptions, dataSize, weightedAveragesOfPrimals, iterCount, miniBatchEnabled), preservesPartitioning = true)
          
      /**
       * Reducer
       */
      val reducedData: (StructSVMModel[X, Y], RDD[(Index, PrimalInfo)], RDD[(Index, BoundedCacheList[Y])], PrimalInfo, Int) = reducer(temp, indexedPrimalsRDD, indexedCacheRDD, globalModel, weightedAveragesOfPrimals, d, beta = 1.0)

      // Update the global model and the primal for each i
      globalModel = reducedData._1
      indexedPrimalsRDD = reducedData._2
      indexedCacheRDD = reducedData._3
      weightedAveragesOfPrimals = reducedData._4
      iterCount = reducedData._5
      
      // println("globalModel.w = %s".format(globalModel.getWeights()(0 to 5).toDenseVector))
      // println("globalModel.ell = %f".format(globalModel.getEll()))

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

      val trainError = SolverUtils.averageLoss(data, lossFn, predictFn, debugModel, dataSize)
      val testError =
        if (solverOptions.testDataRDD.isDefined)
          SolverUtils.averageLoss(solverOptions.testDataRDD.get, lossFn, predictFn, debugModel, dataSize)
        else
          0.00

      val f = -SolverUtils.objectiveFunction(debugModel.getWeights(), debugModel.getEll(), solverOptions.lambda)
      val gapTup = SolverUtils.dualityGap(data, featureFn, lossFn, oracleFn, debugModel, solverOptions.lambda, dataSize)
      val gap = gapTup._1
      val primal = f + gap

      // assert(gap >= 0.0, "Gap is negative")

      if (verboseDebug)
        debugSb ++= "# sum(w): %f, ell: %f\n".format(debugModel.getWeights().sum, debugModel.getEll())

      // logger.info("[DATA] %d,%f,%f,%f\n".format(roundNum, elapsedTime, trainError, testError))
      println("[Round #%d] Train loss = %f, Test loss = %f, Primal = %f, Gap = %f\n".format(roundNum, trainError, testError, primal, gap))
      val curTime = (System.currentTimeMillis() - startTime) / 1000
      debugSb ++= "%d,%d,%f,%f,%f,%f,%f\n".format(roundNum, curTime, primal, f, gap, trainError, testError)
    }

    if (solverOptions.doWeightedAveraging) {
      globalModel.updateWeights(weightedAveragesOfPrimals._1)
      globalModel.updateEll(weightedAveragesOfPrimals._2)
    }

    if (verboseDebug) {
      println("globalModel.weights is sparse - " + globalModel.getWeights.isInstanceOf[SparseVector[Double]])
      println("w_i is sparse - " + indexedPrimalsRDD.first._2._1.isInstanceOf[SparseVector[Double]])
    }
    (globalModel, debugSb.toString())
  }

  /**
   * Takes as input a set of data and builds a SSVM model trained using BCFW
   */
  def mapper(dataIterator: Iterator[(Index, DataShard[X, Y])],
             localModel: StructSVMModel[X, Y],
             featureFn: (X, Y) => Vector[Double], // (y, x) => FeatureVect, 
             lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossVal, 
             oracleFn: (StructSVMModel[X, Y], X, Y) => Y, // (model, y_i, x_i) => Lab, 
             predictFn: (StructSVMModel[X, Y], X) => Y,
             solverOptions: SolverOptions[X, Y],
             dataSize: Int,
             averagedPrimalInfo: PrimalInfo,
             iterCount: Int,
             miniBatchEnabled: Boolean): Iterator[LocalProcessedShard[X, Y]] = {
    
    // println("*mapper begin*")

    val prevModel: StructSVMModel[X, Y] = localModel.clone()

    val numPasses = solverOptions.numPasses
    val lambda = solverOptions.lambda
    val debugOn: Boolean = solverOptions.debug

    val verboseDebug: Boolean = false

    /**
     * Reorganize data for training
     */
    val zippedData: Array[(Index, DataShard[X, Y])] = dataIterator.toArray//.sortBy(_._1)
    val data: Array[LabeledObject[X, Y]] = zippedData.map(x => x._2.data)
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
        case (index, dataShard) => oracleCache(globalToLocal(index)) = dataShard.cache.get
      }

    // Keep track of how many decodings were weak
    var numWeakDecodings = 0
    var totalDecodings = 0

    val maxOracle = oracleFn
    val phi = featureFn
    // Number of dimensions of \phi(x, y)
    val d: Int = localModel.getWeights().size

    // Only to keep track of the \Delta localModel
    val deltaLocalModel = new StructSVMModel(SparseVector.zeros(d), 0.0, SparseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    val eps: Double = 2.2204E-16

    var k: Int = iterCount
    val n: Int = data.size
    val globalN: Int = dataSize

    val wMat: Array[Vector[Double]] = Array.fill(n)(null)
    val prevWMat: Array[Vector[Double]] = Array.fill(n)(null)
    val ellMat: Vector[Double] = Vector.zeros[Double](n)

    // Copy w_i's and l_i's into local wMat and ellMat
    for (globIdx <- globalDataIdx) {
      val i = globalToLocal(globIdx) // localIdx
      wMat(i) = zippedData(i)._2.primalInfo._1
      prevWMat(i) = zippedData(i)._2.primalInfo._1.copy
      ellMat(i) = zippedData(i)._2.primalInfo._2
    }

    val prevEllMat: Vector[Double] = ellMat.copy

    var ell: Double = localModel.getEll()

    if (verboseDebug) {
      println("wMat before pass: " + localModel.getWeights()(0 to 5).toDenseVector)
      println("ellmat before pass: " + ellMat(0 to min(ellMat.size, 5)).toDenseVector)
      println("Ell before pass = " + ell)
    }

    // Initialization in case of Weighted Averaging
    var wAvg: Vector[Double] =
      if (solverOptions.doWeightedAveraging)
        averagedPrimalInfo._1
      else null
    var lAvg: Double =
      if (solverOptions.doWeightedAveraging)
        averagedPrimalInfo._2
      else 0.0

    for ((datapoint, globalIdx) <- data.zip(globalDataIdx)) {

      // Convert globalIdx to localIdx a.k.a "i"
      val i: Index = globalToLocal(globalIdx)
      // val i: Index = globalIdx
      
      //if (globalIdx < 10)
        //println("Index = " + globalIdx)

      // 1) Pick example
      val pattern: X = datapoint.pattern
      val label: Y = datapoint.label

      // 2.a) Search for candidates
      val bestCachedCandidateForI: Option[Y] =
        if (solverOptions.enableOracleCache && oracleCache.contains(i)) {
          val candidates: Seq[(Double, Int)] =
            oracleCache(i)
              .map(y_i => (((phi(pattern, label) - phi(pattern, y_i)) :* (1 / (globalN * lambda))),
                (1.0 / globalN) * lossFn(label, y_i))) // Map each cached y_i to their respective (w_s, ell_s)
              .map {
                case (w_s, ell_s) =>
                  (localModel.getWeights().t * (wMat(i) - w_s) - ((ellMat(i) - ell_s) * (1 / lambda))) /
                    ((wMat(i) - w_s).t * (wMat(i) - w_s) + eps) // Map each (w_s, ell_s) to their respective step-size values	
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
          val ystar = maxOracle(localModel, pattern, label)

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
      totalDecodings += 1

      if (i < 5 && verboseDebug) {
        println("y_star(%d) = %s".format(i, ystar_i))
      }

      // 3) Define the update quantities
      val psi_i: Vector[Double] = phi(pattern, label) - phi(pattern, ystar_i)
      val w_s: Vector[Double] = psi_i :* (1.0 / (globalN * lambda))
      val loss_i: Double = lossFn(label, ystar_i)
      val ell_s: Double = (1.0 / globalN) * loss_i

      // 4) Get step-size gamma
      val gamma: Double =
        if (solverOptions.doLineSearch) {
          val thisModel = if (miniBatchEnabled) prevModel else localModel
          val gamma_opt = (thisModel.getWeights().t * (wMat(i) - w_s) - ((ellMat(i) - ell_s) * (1.0 / lambda))) /
            ((wMat(i) - w_s).t * (wMat(i) - w_s) + eps)
          max(0.0, min(1.0, gamma_opt))
        } else {
          (2.0 * globalN) / (k + 2.0 * globalN)
        }

      if (gamma < 0.5) {
        numWeakDecodings += 1
        // println("[WARN] Weak decoding resulted in gamma = %f, expected gamma >= 0.5.".format(gamma))
        // println("[WARN] %d/%d decodings have been weak so far in this round.".format(numWeakDecodings, totalDecodings))
      }

      // 5, 6, 7, 8) Update the weights of the model
      if (miniBatchEnabled) {
        wMat(i) = wMat(i) * (1.0 - gamma) + (w_s * gamma)
        // wMat(::, i) := wMat(::, i) * (1.0 - gamma) + (w_s * gamma)
        ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma)
        deltaLocalModel.updateWeights(localModel.getWeights() + (wMat(i) - prevWMat(i)))
        deltaLocalModel.updateEll(localModel.getEll() + (ellMat(i) - prevEllMat(i)))
      } else {
        // In case of CoCoA
        val tempWeights1: Vector[Double] = localModel.getWeights() - wMat(i)
        localModel.updateWeights(tempWeights1)
        deltaLocalModel.updateWeights(tempWeights1)
        wMat(i) = wMat(i) * (1.0 - gamma) + (w_s * gamma)
        val tempWeights2: Vector[Double] = localModel.getWeights() + wMat(i)
        localModel.updateWeights(tempWeights2)
        deltaLocalModel.updateWeights(tempWeights2)

        ell = ell - ellMat(i)
        ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma)
        ell = ell + ellMat(i)
      }

      // 9) Optionally update the weighted average
      if (solverOptions.doWeightedAveraging) {
        val rho: Double = 2.0 / (k + 2.0)
        wAvg = (wAvg * (1.0 - rho)) + (localModel.getWeights() * rho)
        lAvg = (lAvg * (1.0 - rho)) + (ell * rho)
      }

      k = k + 1

    }

    if (verboseDebug) {
      println("wMat after pass: " + localModel.getWeights()(0 to 5).toDenseVector)
      println("ellmat after pass: " + ellMat(0 to min(ellMat.size, 5)).toDenseVector)
      println("Ell after pass = " + ell)
    }

    localModel.updateEll(ell)

    val localIndexedDeltaPrimals: Array[(Index, (LocalIndexedPrimalsCache[Y]))] = zippedData.map(_._1).map(k =>
      (k, // Index
        LocalIndexedPrimalsCache(((wMat(globalToLocal(k)) - prevWMat(globalToLocal(k))), // PrimalInfo.w
          ellMat(globalToLocal(k)) - prevEllMat(globalToLocal(k))), // PrimalInfo.ell
          oracleCache.get(globalToLocal(k))))) // Cache

    val localWeightedAverageOfPrimals: PrimalInfo = (wAvg, lAvg)

    // If this flag is set, return only the change in w's
    localModel.updateWeights(localModel.getWeights() - prevModel.getWeights())
    localModel.updateEll(localModel.getEll() - prevModel.getEll())

    val localProcessedData = LocalProcessedShard(localModel, localIndexedDeltaPrimals, deltaLocalModel, localWeightedAverageOfPrimals, k)
    
    //println("*mapper end*")

    // Finally return a single element iterator
    { List.empty[(LocalProcessedShard[X, Y])] :+ localProcessedData }.iterator
  }

  /**
   * Takes as input a number of SVM Models, along with Primal information for each data point, and combines them into a single Model and Primal block
   */
  def reducer( // sc: SparkContext,
    zippedModels: RDD[LocalProcessedShard[X, Y]], // The optimize step returns k blocks. Each block contains (\Delta LocalModel, [\Delta PrimalInfo_i]).
    oldPrimalInfo: RDD[(Index, PrimalInfo)],
    oldCache: RDD[(Index, BoundedCacheList[Y])],
    oldGlobalModel: StructSVMModel[X, Y],
    oldWeightedAveragePrimals: PrimalInfo,
    d: Int,
    beta: Double): (StructSVMModel[X, Y], RDD[(Index, PrimalInfo)], RDD[(Index, BoundedCacheList[Y])], PrimalInfo, Int) = {

    val k: Double = zippedModels.count.toDouble // This refers to the number of localModels generated

    // Here, map is applied k(=#workers) times
    val sumDeltaWeights =
      zippedModels.map(model => model.localModel.getWeights()).reduce((deltaWeightA, deltaWeightB) => deltaWeightA + deltaWeightB)
    val sumDeltaElls =
      zippedModels.map(model => model.localModel.getEll).reduce((ellA, ellB) => ellA + ellB)

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

    val newWeightedAveragePrimals: PrimalInfo =
      if (solverOptions.doWeightedAveraging) {
        /**
         * Repeat the same for the Weighted Averages (wAvg, lAvg)
         */
        val sumNewWeightedAveragePrimals =
          zippedModels.map {
            case localProcessedShard =>
              localProcessedShard.localWeightedAAvgOfPrimals
          }.reduce {
            case ((w1, ell1), (w2, ell2)) =>
              (w1 + w2, ell1 + ell2)
          }

        /* TODO Which one to choose? Thinking assignment for self.
     * val newWeightedAveragePrimals: PrimalInfo = (
      oldWeightedAveragePrimals._1 + (sumNewWeightedAveragePrimals._1 / k) * beta,
      oldWeightedAveragePrimals._2 + (sumNewWeightedAveragePrimals._2 / k) * beta)*/

        ((sumNewWeightedAveragePrimals._1 / k) * beta,
          (sumNewWeightedAveragePrimals._2 / k) * beta)
      } else
        oldWeightedAveragePrimals

    /**
     * Average all the iterCounts and use this as the starting point for 'k' in the next mapper step
     */
    val newIterCount = (zippedModels.map {
      case localProcessedShard =>
        localProcessedShard.k
    }
      .reduce((x, y) => x + y) / k).round.toInt

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
      case localProcessedShard =>
        localProcessedShard.localIndexedDeltaPrimals
    }

    val indexedPrimals: RDD[(Index, PrimalInfo)] =
      primalsAndCache.mapValues { case localIndexedPrimalsCache => localIndexedPrimalsCache.primalInfo }
        .rightOuterJoin(oldPrimalInfo)
        .mapValues {
          case (Some((newW, newEll)), (prevW, prevEll)) =>
            (prevW + (newW * (beta / k)),
              prevEll + (newEll * (beta / k)))
          case (None, (prevW, prevEll)) => (prevW, prevEll)
        }

    val indexedCache =
      if (solverOptions.enableOracleCache)
        primalsAndCache.mapValues { case localIndexedPrimalsCache => localIndexedPrimalsCache.cache }
          .rightOuterJoin(oldCache)
          .mapValues {
            // newCache includes entries in the previous cache too
            case (Some(newCache), oldCache) => newCache.get
            case (None, oldCache)           => oldCache
          }
      else
        oldCache

    // indexedPrimals isn't materialized till an RDD action is called. Force this by calling one.
    indexedPrimals.checkpoint()
    indexedPrimals.count()
    indexedCache.count()

    (newGlobalModel, indexedPrimals, indexedCache, newWeightedAveragePrimals, newIterCount)
  }

}