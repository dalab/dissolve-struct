package ch.ethz.dal.dbcfw.optimization

import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import org.apache.spark.SparkContext
import ch.ethz.dal.dbcfw.classification.Types._

object DBCFWSolver {

  /**
   * Takes as input a set of data and builds a SSVM model trained using BCFW
   */
  def optimizeCoCoA(dataIterator: Iterator[(Index, (LabeledObject, Primal))],
    localModel: StructSVMModel,
    featureFn: (Vector[Double], Matrix[Double]) => Vector[Double], // (y, x) => FeatureVect, 
    lossFn: (Vector[Double], Vector[Double]) => Double, // (yTruth, yPredict) => LossVal, 
    oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double], // (model, y_i, x_i) => Lab, 
    predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
    solverOptions: SolverOptions,
    returnModelDiff: Boolean,
    returnPrimalDiff: Boolean,
    miniBatchEnabled: Boolean): Iterator[(StructSVMModel, Array[(Index, Primal)])] = {

    val prevModel: StructSVMModel = localModel.clone()

    val numPasses = solverOptions.numPasses
    val lambda = solverOptions.lambda
    val debugOn: Boolean = solverOptions.debug
    val xldebug: Boolean = solverOptions.xldebug

    /**
     * Reorganize data for training
     */
    val zippedData: Array[(Index, (LabeledObject, Primal))] = dataIterator.toArray
    val data: Array[LabeledObject] = zippedData.map(x => x._2._1)
    // Mapping of indexMapping(localIndex) -> globalIndex
    val indexMapping: Array[Int] = zippedData.map(x => x._1).toArray // Creates a mapping where j = indexMapping(i) refers to i-th local (xi, yi) and j-th global (xj, yj)

    val maxOracle = oracleFn
    val phi = featureFn
    // Number of dimensions of \phi(x, y)
    val d: Int = localModel.getWeights().size

    val eps: Double = 2.2204E-16

    var k: Int = 0
    val n: Int = data.size

    val wMat: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, n)
    val ellMat: DenseVector[Double] = DenseVector.zeros[Double](n)

    // Copy w_i's and l_i's into local wMat and ellMat
    for (i <- 0 until n) {
      wMat(::, i) := zippedData(i)._2._2._1
      ellMat(i) = zippedData(i)._2._2._2
    }
    val prevWMat: DenseMatrix[Double] = wMat.copy
    val prevEllMat: DenseVector[Double] = ellMat.copy

    var ell: Double = localModel.getEll()

    // Initialization in case of Weighted Averaging
    var wAvg: DenseVector[Double] =
      if (solverOptions.doWeightedAveraging)
        DenseVector.zeros(d)
      else null
    var lAvg: Double = 0.0

    for ((datapoint, i) <- data.zipWithIndex) {
      // 1) Pick example
      val pattern: Matrix[Double] = data(i).pattern
      val label: Vector[Double] = data(i).label

      // 2) Solve loss-augmented inference for point i
      val ystar_i: Vector[Double] =
        if (!miniBatchEnabled)
          maxOracle(localModel, label, pattern)
        else
          maxOracle(prevModel, label, pattern)

      // 3) Define the update quantities
      val psi_i: Vector[Double] = phi(label, pattern) - phi(ystar_i, pattern)
      val w_s: Vector[Double] = psi_i :* (1 / (n * lambda))
      val loss_i: Double = lossFn(label, ystar_i)
      val ell_s: Double = (1.0 / n) * loss_i

      // 4) Get step-size gamma
      val gamma: Double =
        if (solverOptions.doLineSearch) {
          val thisModel = if (miniBatchEnabled) prevModel else localModel
          val gamma_opt = (thisModel.getWeights().t * (wMat(::, i) - w_s) - ((ellMat(i) - ell_s) * (1 / lambda))) /
            ((wMat(::, i) - w_s).t * (wMat(::, i) - w_s) + eps)
          max(0.0, min(1.0, gamma_opt))
        } else {
          (2.0 * n) / (k + 2.0 * n)
        }

      // 5, 6, 7, 8) Update the weights of the model
      val tempWeights1: Vector[Double] = localModel.getWeights() - wMat(::, i)
      localModel.updateWeights(tempWeights1)
      wMat(::, i) := (wMat(::, i) * (1.0 - gamma)) + (w_s * gamma)
      val tempWeights2: Vector[Double] = localModel.getWeights() + wMat(::, i)
      localModel.updateWeights(tempWeights2)

      ell = ell - ellMat(i)
      ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma)
      ell = ell + ellMat(i)

      // 9) Optionally update the weighted average
      if (solverOptions.doWeightedAveraging) {
        val rho: Double = 2.0 / (k + 2.0)
        wAvg = (wAvg * (1.0 - rho)) + (localModel.getWeights * rho)
        lAvg = (lAvg * (1.0 - rho)) + (ell * rho)
      }
    }

    if (solverOptions.doWeightedAveraging) {
      localModel.updateWeights(wAvg)
      localModel.updateEll(lAvg)
    } else {
      localModel.updateEll(ell)
    }

    // If this flag is set, return only the change in w's
    if (returnModelDiff) {
      localModel.updateWeights(localModel.getWeights() - prevModel.getWeights())
      localModel.updateEll(localModel.getEll() - prevModel.getEll())
    }

    val localIndexedPrimals: Array[(Index, Primal)] = zippedData.map(x => x._1).zip(
      if (returnPrimalDiff)
        for (k <- (0 until n).toList) yield (wMat(::, k), ellMat(k))
      else
        for (k <- (0 until n).toList) yield (wMat(::, k) - prevWMat(::, k), ellMat(k) - prevEllMat(k)))

    // Finally return a single element iterator
    { List.empty[(StructSVMModel, Array[(Index, Primal)])] :+ (localModel, localIndexedPrimals) }.iterator
  }

  def combineModelsCoCoA( // sc: SparkContext,
    zippedModels: RDD[(StructSVMModel, Array[(Index, Primal)])],
    oldGlobalModel: StructSVMModel,
    d: Int,
    betaByK: Double,
    miniBatchEnabled: Boolean): (StructSVMModel, RDD[(Index, Primal)]) = {

    val numModels: Long = zippedModels.count

    val sumWeights =
      if (!miniBatchEnabled)
        zippedModels.map(model => model._1.getWeights()).reduce((weightA, weightB) => weightA + weightB)
      else
        zippedModels.flatMap(item => item._2).map(indPrimal => indPrimal._2._1).reduce((weightA, weightB) => weightA + weightB)
    val sumElls =
      if (!miniBatchEnabled)
        zippedModels.map(model => model._1.getEll).reduce((ellA, ellB) => ellA + ellB)
      else
        zippedModels.flatMap(item => item._2).map(indPrimal => indPrimal._2._2).reduce((ellA, ellB) => ellA + ellB)

    /**
     * Create the new global model
     */
    val sampleModel = zippedModels.first._1
    val newGlobalModel = new StructSVMModel(oldGlobalModel.getWeights() + (sumWeights / numModels.toDouble) * betaByK,
      oldGlobalModel.getEll() + (sumElls / numModels) * betaByK,
      DenseVector.zeros(d),
      sampleModel.featureFn,
      sampleModel.lossFn,
      sampleModel.oracleFn,
      sampleModel.predictFn)

    /**
     * Merge all the w_i's and l_i's
     */
    val indexedPrimals: RDD[(Index, Primal)] =
      if (!miniBatchEnabled)
        zippedModels.flatMap(x => x._2)
      else null // In case minibatch is enabled, new Primals are obtained through a different pipeline

    (newGlobalModel, indexedPrimals)
  }

  def bcfwOptimizeMiniBatch(dataIterator: Iterator[(LabeledObject, (Vector[Double], Double))],
    model: StructSVMModel,
    featureFn: (Vector[Double], Matrix[Double]) => Vector[Double], // (y, x) => FeatureVect, 
    lossFn: (Vector[Double], Vector[Double]) => Double, // (yTruth, yPredict) => LossVal, 
    oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double], // (model, y_i, x_i) => Lab, 
    predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
    solverOptions: SolverOptions): Iterator[(LabeledObject, (DenseVector[Double], Double))] = {

    val numPasses = solverOptions.numPasses
    val lambda = solverOptions.lambda
    val debugOn: Boolean = solverOptions.debug
    val xldebug: Boolean = solverOptions.xldebug

    val zippedData: Array[(LabeledObject, (Vector[Double], Double))] = dataIterator.toArray

    val maxOracle = oracleFn
    val phi = featureFn
    // Number of dimensions of \phi(x, y)
    val d: Int = model.getWeights().size

    val eps: Double = 2.2204E-16
    val lossWriterFileName = "data/debug/dbcfw-loss.csv"

    var k: Int = 0
    val n: Int = zippedData.size

    // Reconstruct wMat from w_i's obtained
    val wMat: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, n)
    for (i <- 0 until n) {
      wMat(::, i) := zippedData(i)._2._1
    }
    val wMat_prev = wMat.copy

    var ell: Double = model.getEll()

    // Reconstruct ellMat from the ell_i's obtained 
    val ellMat: DenseVector[Double] = DenseVector.zeros[Double](n)
    for (i <- 0 until n) {
      ellMat(i) = zippedData(i)._2._2
    }
    val ellMat_prev = ellMat.copy

    // for (passNum <- 0 until 5) {

    // val i = scala.util.Random.nextInt(n)
    // val datapoint = zippedData(i)

    for ((datapoint, i) <- zippedData.zipWithIndex) {
      // 1) Pick example
      val pattern: Matrix[Double] = datapoint._1.pattern
      val label: Vector[Double] = datapoint._1.label

      // 2) Solve loss-augmented inference for point i
      val ystar_i: Vector[Double] = maxOracle(model, label, pattern)

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

      // 5, 6) Compute w_i^{(k+1)} and l_i^{(k+1)}
      wMat(::, i) := (wMat(::, i) * (1.0 - gamma)) + (w_s * gamma) // Obtain w_i^{(k+1)}
      ellMat(i) = (ellMat(i) * (1.0 - gamma)) + (ell_s * gamma) // Obtain l_i^{(k+1)}
    }
    // }

    // Finally return an iterator of datapoints zipped with w_i's and ell_i's
    val newLocalData: List[(LabeledObject, (DenseVector[Double], Double))] = for (i <- (0 until n).toList) yield {
      val diffW: DenseVector[Double] = wMat(::, i).toDenseVector
      val diffL: Double = ellMat(i)
      (zippedData(i)._1, (diffW, diffL))
    }
    newLocalData.toIterator
  }

  def bcfwCombine(prevModel: StructSVMModel,
    zippedTrainedData: RDD[(LabeledObject, (DenseVector[Double], Double))]): StructSVMModel = {

    val zippedArray = zippedTrainedData.toArray()

    val numModels: Long = zippedTrainedData.count
    val d: Int = zippedTrainedData.first._2._1.size
    // TODO Convert into an aggregate function

    // val sumDiffWeights: DenseVector[Double] = zippedTrainedData.map(tup => tup._2).reduce((x, y) => x + y)
    // val sumDiffElls: Double = zippedTrainedData.map(tup => tup._3).reduce((x, y) => x + y)

    val sumDiffs: (Vector[Double], Double) = zippedTrainedData.map(kv => kv._2).reduce((x, y) => (x._1 + y._1, x._2 + y._2))

    new StructSVMModel(prevModel.getWeights() + sumDiffs._1,
      prevModel.getEll() + sumDiffs._2,
      DenseVector.zeros(d),
      prevModel.featureFn,
      prevModel.lossFn,
      prevModel.oracleFn,
      prevModel.predictFn)
  }

  /**
   * Takes as input a set of data and builds a SSVM model trained using BCFW
   */
  def optimize(dataIterator: Iterator[LabeledObject],
    model: StructSVMModel,
    featureFn: (Vector[Double], Matrix[Double]) => Vector[Double], // (y, x) => FeatureVect, 
    lossFn: (Vector[Double], Vector[Double]) => Double, // (yTruth, yPredict) => LossVal, 
    oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double], // (model, y_i, x_i) => Lab, 
    predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
    solverOptions: SolverOptions,
    returnDiff: Boolean): Iterator[StructSVMModel] = {

    val prevModel: StructSVMModel = model.clone()

    val numPasses = solverOptions.numPasses
    val lambda = solverOptions.lambda
    val debugOn: Boolean = solverOptions.debug
    val xldebug: Boolean = solverOptions.xldebug

    val data: Array[LabeledObject] = dataIterator.toArray

    val maxOracle = oracleFn
    val phi = featureFn
    // Number of dimensions of \phi(x, y)
    // val d: Int = phi(data(0).label, data(0).pattern).size
    val d: Int = model.getWeights().size

    val eps: Double = 2.2204E-16
    val lossWriterFileName = "data/debug/dbcfw-loss.csv"

    var k: Int = 0
    val n: Int = data.size
    val wMat: DenseMatrix[Double] = DenseMatrix.zeros[Double](d, n)
    var ell: Double = 0.0
    val ellMat: DenseVector[Double] = DenseVector.zeros[Double](n)

    val debugModel: StructSVMModel = model.clone()

    // Initialization in case of Weighted Averaging
    var wAvg: DenseVector[Double] =
      if (solverOptions.doWeightedAveraging)
        DenseVector.zeros(d)
      else null
    var lAvg: Double = 0.0

    for (passNum <- 0 until solverOptions.numPasses) {
      for ((datapoint, i) <- data.zipWithIndex) {
        // 1) Pick example
        val pattern: Matrix[Double] = data(i).pattern
        val label: Vector[Double] = data(i).label

        // 2) Solve loss-augmented inference for point i
        val ystar_i: Vector[Double] = maxOracle(model, label, pattern)

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
      }
    }

    if (solverOptions.doWeightedAveraging) {
      model.updateWeights(wAvg)
      model.updateEll(lAvg)
    } else {
      model.updateEll(ell)
    }

    // If this flag is set, return only the change in w's
    if (returnDiff) {
      model.updateWeights(model.getWeights() - prevModel.getWeights())
      model.updateEll(model.getEll() - prevModel.getEll())
    }

    // Finally return a single element iterator
    { List.empty[StructSVMModel] :+ model }.toIterator
  }

  /**
   * Combines multiple StructSVMModels into a single StructSVMModel
   * Works by taking the average of Weights and Ells over multiple models
   */
  def combineModels( // sc: SparkContext,
    models: RDD[StructSVMModel]): StructSVMModel = {
    val numModels: Long = models.count
    val d: Int = models.first.getWeights.size
    // TODO Convert into an aggregate function
    val sumWeights = models.map(model => model.getWeights).reduce((weightA, weightB) => weightA + weightB).toDenseVector
    val sumElls = models.map(model => model.getEll).reduce((ellA, ellB) => ellA + ellB)

    /*val zeroModel: StructSVMModel = new StructSVMModel(DenseVector.zeros(d),
      0.0,
      DenseVector.zeros(d),
      models.first.featureFn,
      models.first.lossFn,
      models.first.oracleFn,
      models.first.predictFn)*/

    new StructSVMModel(sumWeights / numModels.toDouble,
      sumElls / numModels,
      DenseVector.zeros(d),
      models.first.featureFn,
      models.first.lossFn,
      models.first.oracleFn,
      models.first.predictFn)
  }

  def main(args: Array[String]): Unit = {}

}