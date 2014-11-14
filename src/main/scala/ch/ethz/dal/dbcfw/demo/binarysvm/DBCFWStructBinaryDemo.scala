/**
 *
 */
package ch.ethz.dal.dbcfw.demo.binarysvm

import scala.Array.canBuildFrom
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Matrix
import breeze.linalg.Vector
import breeze.linalg.max
import breeze.numerics.abs
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.optimization.SolverOptions
import ch.ethz.dal.dbcfw.classification.StructSVMWithBCFW
import ch.ethz.dal.dbcfw.classification.StructSVMWithSSG
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import ch.ethz.dal.dbcfw.classification.StructSVMWithDBCFW
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import ch.ethz.dal.dbcfw.utils.DissolveUtils

/**
 *
 */
object DBCFWStructBinaryDemo {

  val debugOn: Boolean = true

  /**
   * Feature function
   *
   * Analogous to phi(y) in (2)
   * Returns y_i * x_i
   */
  def featureFnBin(y: Double, x: Vector[Double]): Vector[Double] =
    x :* y

  /**
   * Represent feature vector as 1xn Matrix
   */
  def featureFn(y: Vector[Double], x: Matrix[Double]): Vector[Double] =
    x.toDenseMatrix.toDenseVector :* y(0)

  /**
   * Loss function
   *
   * Returns 0 if yTruth == yPredict, 1 otherwise
   * Equivalent to max(0, 1 - y w^T x)
   */
  def lossFnBin(yTruth: Double, yPredict: Double): Double =
    abs(yTruth - yPredict)

  def lossFn(yTruth: Vector[Double], yPredict: Vector[Double]): Double =
    if (yTruth(0) == yPredict(0))
      0.0
    else 1.0

  /**
   * Maximization Oracle
   *
   * Want: max L(y_i, y) - <w, psi_i(y)>
   * This returns a negative value in case of a correct prediction and positive in case of a false prediction
   */
  def oracleFnBin(weights: Vector[Double], yi: Double, xi: Vector[Double]): Double =
    lossFnBin(yi, weights dot xi) - (weights dot featureFnBin(yi, xi))

  def oracleFn(model: StructSVMModel[Matrix[Double], Vector[Double]], yi: Vector[Double], xi: Matrix[Double]): Vector[Double] =
    DenseVector.fill(1) { lossFn(yi, DenseVector.fill(1) { model.getWeights().toDenseVector dot xi.toDenseMatrix.toDenseVector }) - (model.getWeights() dot featureFn(yi, xi)) }

  /**
   * Prediction function
   */
  def predictFnBin(weights: Vector[Double], xi: Vector[Double]): Double =
    if ((weights dot featureFnBin(1.0, xi)) > (weights dot featureFnBin(-1.0, xi)))
      1.0
    else
      -1.0

  def predictFn(model: StructSVMModel[Matrix[Double], Vector[Double]], xi: Matrix[Double]): Vector[Double] =
    if ((model.getWeights() dot featureFn(DenseVector.ones[Double](1), xi)) > (model.getWeights() dot featureFn(DenseVector.fill[Double](1) { -1.0 }, xi)))
      DenseVector.ones[Double](1)
    else
      DenseVector.fill[Double](1) { -1.0 }

  def loadLibSVMFile(filename: String): Vector[LabeledObject[Matrix[Double], Vector[Double]]] = {
    var n: Int = 0
    var ndims: Int = 0

    if (debugOn)
      println("Obtaining data metadata")
    // First pass, get number of data points and number of features
    for (line <- scala.io.Source.fromFile(filename).getLines()) {
      n += 1
      ndims = max(ndims, max(line.split(" ").slice(1, line.split(" ").size).map(s => s.split(":")(0) toInt)))
    }

    // Create a n-dim vector, each element containing a 1-dim vector
    // val labels: DenseVector[DenseVector[Double]] = DenseVector.fill(n) { DenseVector.zeros(1) }
    // Create a n-dim vector, each element containing a 1xndim dimensional matrix
    // val patterns: DenseVector[DenseMatrix[Double]] = DenseVector.fill(n) { DenseMatrix.zeros(1, ndims) }
    // Create LabeledObject s

    if (debugOn)
      println("Obtained data with %d datapoints, and ndims %d".format(n, ndims))

    // Second pass - Fill in data
    val data: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = DenseVector.fill(n) { new LabeledObject(DenseVector.zeros(1), DenseMatrix.zeros(1, ndims)) }
    if (debugOn)
      println("Creating LabeledObject Vector with size %d, pattern=%dx%d, label=%dx1".format(data.size, data(0).pattern.rows, data(0).pattern.cols, data(0).label.size))
    var row: Int = 0
    for (line <- scala.io.Source.fromFile(filename).getLines()) {
      // Convert one line in libSVM format to a string array
      val content: Array[String] = line.split(" ")
      // Read 1st column (Label)
      data(row).label(0) = if (content(0).contentEquals("+1")) 1 else -1

      // Read rest of the columns (Patterns)
      for (ele <- content.slice(1, content.size)) {
        val col = (ele.split(":")(0) toInt) - 1
        val item = ele.split(":")(1) toDouble

        // println("row=%d column=%d".format(row, col))
        data(row).pattern(0, col) = item
      }
      row += 1
    }

    data
  }

  /**
   * BSVM - DBCFW style
   */
  def dbcfwBSVM(libSvmFilePath: String): Unit = {
    // val data: Vector[LabeledObject] = loadLibSVMFile(libSvmFilePath)
    val data: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = DissolveUtils.loadLibSVMBinaryFileHD(libSvmFilePath, sparse = true, labelMap = Map("+1" -> 1, "-1" -> -1))
    println("Loaded data with %d rows, pattern=%dx%d, label=%dx1".format(data.size, data(0).pattern.rows, data(0).pattern.cols, data(0).label.size))

    // Fix seed for reproducibility
    util.Random.setSeed(1)

    // Split data into training and test datasets
    val trnPrc = 0.8
    val perm: List[Int] = util.Random.shuffle((0 until data.size) toList)
    val cutoffIndex: Int = (trnPrc * perm.size) toInt
    val train_data = data(perm.slice(0, cutoffIndex)).toArray // Obtain in range [0, cutoffIndex)
    val test_data = data(perm.slice(cutoffIndex, perm.size)).toArray // Obtain in range [cutoffIndex, data.size)

    val solverOptions: SolverOptions[Matrix[Double], Vector[Double]] = new SolverOptions()

    solverOptions.numPasses = 3 // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = false
    solverOptions.xldebug = false
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = false
    solverOptions.testData = Some(test_data)

    solverOptions.sampleWithReplacement = false
    solverOptions.NUM_PART = 1
    solverOptions.autoconfigure = false

    /*val trainer: StructSVMWithSSG = new StructSVMWithSSG(train_data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)*/

    /*val trainer: StructSVMWithBCFW = new StructSVMWithBCFW(train_data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)*/

    solverOptions.NUM_PART = 2
    solverOptions.sample = "frac"
    solverOptions.sampleFrac = 0.5

    val conf = new SparkConf().setAppName("Binary-DBCFW").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    val training = sc.parallelize(train_data)

    val trainer: StructSVMWithDBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithDBCFW[Matrix[Double], Vector[Double]](
      training,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)

    val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

    var truePredictions = 0
    val totalPredictions = test_data.size

    for (item <- test_data) {
      val prediction = model.predictFn(model, item.pattern)
      if (prediction(0) == item.label(0))
        truePredictions += 1
    }

    println("Accuracy on Test set = %d/%d = %.4f".format(truePredictions,
      totalPredictions,
      (truePredictions.toDouble / totalPredictions.toDouble) * 100))
  }

  /**
   * MLLib implementation of SVMSGD, on LibSVM format data
   */
  def mllibBSVM(libSvmFilePath: String): Unit = {

    import org.apache.spark.SparkContext
    import org.apache.spark.mllib.classification.SVMWithSGD
    import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
    import org.apache.spark.mllib.regression.LabeledPoint
    import org.apache.spark.mllib.linalg.Vectors
    import org.apache.spark.mllib.util.MLUtils

    val conf = new SparkConf().setAppName("Binary-MLLib")
    val sc = new SparkContext(conf)

    val data = MLUtils.loadLibSVMFile(sc, libSvmFilePath)

    // Split data into training (80%) and test (20%).
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1)
    val training = splits(0).cache()
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
    val model = SVMWithSGD.train(training, numIterations)

    // Compute raw scores on the test set. 
    val scoreAndLabels = test.map { point =>
      val score = model.predict(point.features)
      (score, point.label)
    }

    // Get evaluation metrics.
    val metrics = new BinaryClassificationMetrics(scoreAndLabels)
    val auROC = metrics.areaUnderROC()

    println("Area under ROC = " + auROC)

    val trainError = training.map { point =>
      val score = model.predict(point.features)
      score == point.label
    }.collect().toList.count(_ == true).toDouble / training.count().toDouble

    val testError = test.map { point =>
      val score = model.predict(point.features)
      score == point.label
    }.collect().toList.count(_ == true).toDouble / test.count().toDouble

    println("Training accuracy = " + trainError)
    println("Test accuracy = " + testError)

  }

  def main(args: Array[String]): Unit = {

    // Runs MLLib's SVMSGD on Adult dataset
    // mllibBSVM("data/a1a_mllib.txt")

    // Runs DBCFW on Adult dataset
    dbcfwBSVM("/Users/tribhu/git/DBCFWstruct/data/a1a.txt")
  }

}