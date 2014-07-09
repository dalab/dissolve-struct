package ch.ethz.dal.dbcfw.optimization

import org.apache.spark.rdd.RDD
import breeze.linalg._
import breeze.numerics._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import org.apache.spark.SparkContext

object DBCFWSolver {

  /**
   * Takes as input a set of data and builds a SSVM model trained using BCFW
   */
  def optimize( // sc: SparkContext,
    dataIterator: Iterator[LabeledObject],
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