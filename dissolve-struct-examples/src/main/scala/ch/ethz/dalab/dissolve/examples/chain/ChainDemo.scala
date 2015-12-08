package ch.ethz.dalab.dissolve.examples.chain

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Matrix
import breeze.linalg.Vector
import breeze.linalg.argmax
import breeze.linalg.max
import breeze.linalg.sum
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.classification.StructSVMWithBCFW
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
import ch.ethz.dalab.dissolve.examples.utils.ExampleUtils
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.optimization.GapThresholdCriterion
import ch.ethz.dalab.dissolve.optimization.RoundLimitCriterion
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import ch.ethz.dalab.dissolve.optimization.TimeLimitCriterion
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.utils.cli.CLAParser
import ch.ethz.dalab.dissolve.optimization.DistBCFW
import ch.ethz.dalab.dissolve.models.LinearChainCRF
import ch.ethz.dalab.dissolve.optimization.DistributedSolver

/**
 * How to generate the input data:
 * While in the data directory, run
 * python convert-ocr-data.py
 *
 */
object ChainDemo {

  /**
   * Reads data produced by the convert-ocr-data.py script and loads into memory as a vector of Labeled objects
   *
   *  TODO
   *  * Take foldNumber as a parameter and return training and test set
   */
  def loadData(patternsFilename: String, labelsFilename: String, foldFilename: String): Array[LabeledObject[Matrix[Double], Vector[Double]]] = {
    val patterns: Array[String] = scala.io.Source.fromFile(patternsFilename).getLines().toArray[String]
    val labels: Array[String] = scala.io.Source.fromFile(labelsFilename).getLines().toArray[String]
    val folds: Array[String] = scala.io.Source.fromFile(foldFilename).getLines().toArray[String]

    val n = labels.size

    assert(patterns.size == labels.size, "#Patterns=%d, but #Labels=%d".format(patterns.size, labels.size))
    assert(patterns.size == folds.size, "#Patterns=%d, but #Folds=%d".format(patterns.size, folds.size))

    val data: Array[LabeledObject[Matrix[Double], Vector[Double]]] = Array.fill(n) { null }

    for (i <- 0 until n) {
      // Expected format: id, #rows, #cols, (pixels_i_j,)* pixels_n_m
      val patLine: List[Double] = patterns(i).split(",").map(x => x.toDouble) toList
      // Expected format: id, #letters, (letters_i)* letters_n
      val labLine: List[Double] = labels(i).split(",").map(x => x.toDouble) toList

      val patNumRows: Int = patLine(1) toInt
      val patNumCols: Int = patLine(2) toInt
      val labNumEles: Int = labLine(1) toInt

      assert(patNumCols == labNumEles, "pattern_i.cols == label_i.cols violated in data")

      val patVals: Array[Double] = patLine.slice(3, patLine.size).toArray[Double]
      // The pixel values should be Column-major ordered
      val thisPattern: DenseMatrix[Double] = DenseVector(patVals).toDenseMatrix.reshape(patNumRows, patNumCols)

      val labVals: Array[Double] = labLine.slice(2, labLine.size).toArray[Double]
      assert(List.fromArray(labVals).count(x => x < 0 || x > 26) == 0, "Elements in Labels should be in the range [0, 25]")
      val thisLabel: DenseVector[Double] = DenseVector(labVals)

      assert(thisPattern.cols == thisLabel.size, "pattern_i.cols == label_i.cols violated in Matrix representation")

      data(i) = new LabeledObject(thisLabel, thisPattern)

    }

    data
  }

  /**
   * ****************************************************************
   *    ___   _____ ____ _      __
   *   / _ ) / ___// __/| | /| / /
   *  / _  |/ /__ / _/  | |/ |/ /
   * /____/ \___//_/    |__/|__/
   *
   * ****************************************************************
   */
  def chainBCFW(): Unit = {

    val PERC_TRAIN: Double = 0.05 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val dataDir: String = "../data/generated";

    val train_data: Array[LabeledObject[Matrix[Double], Vector[Double]]] =
      loadData(dataDir + "/patterns_train.csv", dataDir + "/labels_train.csv", dataDir + "/folds_train.csv")
    val test_data: Array[LabeledObject[Matrix[Double], Vector[Double]]] =
      loadData(dataDir + "/patterns_test.csv", dataDir + "/labels_test.csv", dataDir + "/folds_test.csv")

    println("Running chainBCFW (single worker). Loaded %d training examples, pattern:%dx%d and labels:%dx1"
      .format(train_data.size,
        train_data(0).pattern.rows,
        train_data(0).pattern.cols,
        train_data(0).label.size))

    val solverOptions: SolverOptions[Matrix[Double], Vector[Double]] = new SolverOptions()
    solverOptions.roundLimit = 5
    solverOptions.debug = true
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debug = true
    solverOptions.testData = Some(test_data)

    solverOptions.enableOracleCache = false
    solverOptions.oracleCacheSize = 10

    solverOptions.debugInfoPath = "../debug/debug-bcfw-%d.csv".format(System.currentTimeMillis())

    /*val trainer: StructSVMWithSSG = new StructSVMWithSSG(train_data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)*/

    val crfModel = new LinearChainCRF()

    val trainer: StructSVMWithBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithBCFW[Matrix[Double], Vector[Double]](train_data,
      crfModel,
      solverOptions)

    val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

    var avgTrainLoss: Double = 0.0
    for (item <- train_data) {
      val prediction = model.predict(item.pattern)
      avgTrainLoss += crfModel.lossFn(item.label, prediction)
    }
    println("Average loss on training set = %f".format(avgTrainLoss / train_data.size))

    var avgTestLoss: Double = 0.0
    for (item <- test_data) {
      val prediction = model.predict(item.pattern)
      avgTestLoss += crfModel.lossFn(item.label, prediction)
    }
    println("Average loss on test set = %f".format(avgTestLoss / test_data.size))

  }

  def chainDBCFWSolver(args: Array[String]): Unit = {

    /**
     * Load all options
     */
    val (solverOptions, kwargs) = CLAParser.argsToOptions[Matrix[Double], Vector[Double]](args)
    val dataDir = kwargs.getOrElse("input_path", "../data/generated")
    val appname = kwargs.getOrElse("appname", "chain")
    val debugPath = kwargs.getOrElse("debug_file", "chain-%d.csv".format(System.currentTimeMillis() / 1000))
    solverOptions.debugInfoPath = debugPath

    /**
     * Set and configure Spark
     */
    val conf = new SparkConf().setAppName(appname).setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    /**
     * Load Data
     */
    val trainData: Array[LabeledObject[Matrix[Double], Vector[Double]]] =
      loadData(dataDir + "/patterns_train.csv", dataDir + "/labels_train.csv", dataDir + "/folds_train.csv")
    val testData: Array[LabeledObject[Matrix[Double], Vector[Double]]] =
      loadData(dataDir + "/patterns_test.csv", dataDir + "/labels_test.csv", dataDir + "/folds_test.csv")

    val testDataRDD =
      if (solverOptions.enableManualPartitionSize)
        Some(sc.parallelize(testData, solverOptions.NUM_PART))
      else
        Some(sc.parallelize(testData))

    val trainDataRDD =
      if (solverOptions.enableManualPartitionSize)
        sc.parallelize(trainData, solverOptions.NUM_PART)
      else
        sc.parallelize(trainData)

    /**
     * Train model
     */
    val crfModel = new LinearChainCRF(disablePairwise = false, useBPDecoding = false)

    val solver: DistributedSolver[Matrix[Double], Vector[Double]] =
      new DistBCFW(crfModel, solverOptions)

    println("Running Distributed BCFW with CoCoA. Loaded data with %d rows, pattern=%dx%d, label=%dx1"
      .format(trainData.size, trainData(0).pattern.rows, trainData(0).pattern.cols, trainData(0).label.size))

    val model: StructSVMModel[Matrix[Double], Vector[Double]] = solver.train(trainDataRDD, testDataRDD)

    /**
     * Post-training statistics
     */
    var avgTrainLoss: Double = 0.0
    for (item <- trainData) {
      val prediction = model.predict(item.pattern)
      avgTrainLoss += crfModel.lossFn(item.label, prediction)
    }
    println("Average loss on training set = %f".format(avgTrainLoss / trainData.size))

    var avgTestLoss: Double = 0.0
    for (item <- testData) {
      val prediction = model.predict(item.pattern)
      avgTestLoss += crfModel.lossFn(item.label, prediction)
    }
    println("Average loss on test set = %f".format(avgTestLoss / testData.size))

  }

  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("conf/log4j.properties")

    System.setProperty("spark.akka.frameSize", "512")

    chainDBCFWSolver(args)

    // chainBCFW()
  }

}