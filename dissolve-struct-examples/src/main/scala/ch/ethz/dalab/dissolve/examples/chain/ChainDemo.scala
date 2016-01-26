package ch.ethz.dalab.dissolve.examples.chain

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Matrix
import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.models.LinearChainCRF
import ch.ethz.dalab.dissolve.optimization.DistBCFW
import ch.ethz.dalab.dissolve.optimization.DistSSGD
import ch.ethz.dalab.dissolve.optimization.DistributedSolver
import ch.ethz.dalab.dissolve.optimization.LocalSSGD
import ch.ethz.dalab.dissolve.optimization.SSVMClassifier
import ch.ethz.dalab.dissolve.regression.LabeledObject

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

    val dataDir: String = "../data/generated"

    val train_data: Array[LabeledObject[Matrix[Double], Vector[Double]]] =
      loadData(dataDir + "/patterns_train.csv", dataDir + "/labels_train.csv", dataDir + "/folds_train.csv")
    val test_data: Array[LabeledObject[Matrix[Double], Vector[Double]]] =
      loadData(dataDir + "/patterns_test.csv", dataDir + "/labels_test.csv", dataDir + "/folds_test.csv")

    println("Running chainBCFW (single worker). Loaded %d training examples, pattern:%dx%d and labels:%dx1"
      .format(train_data.size,
        train_data(0).pattern.rows,
        train_data(0).pattern.cols,
        train_data(0).label.size))

    val crfModel = new LinearChainCRF(26, disablePairwise = false, useBPDecoding = false)

    // val solver = new LocalBCFW(crfModel, numPasses = 100, debug = true, debugMultiplier = 0, gapThreshold = 0.1, gapCheck = 0, timeBudget = 1)
    val solver = new LocalSSGD(crfModel, numPasses = 100, debug = true, debugMultiplier = 0)

    val classifier = new SSVMClassifier(crfModel)

    classifier.train(train_data, test_data, solver)

    val x = train_data(0).pattern
    val y = classifier.predict(x)

  }

  def chainDBCFWSolver(): Unit = {

    val dataDir = "../data/generated"
    val appname = "chain"
    val debugPath = "chain-%d.csv".format(System.currentTimeMillis() / 1000)
    val numPartitions = 6
    val numStates = 26

    /**
     * Set and configure Spark
     */
    val conf = new SparkConf().setAppName(appname).setMaster("local[3]")
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
      sc.parallelize(testData, numPartitions)

    val trainDataRDD =
      sc.parallelize(trainData, numPartitions)

    /**
     * Train model
     */
    val crfModel = new LinearChainCRF(numStates, disablePairwise = false, useBPDecoding = false)

    /*val solver: DistributedSolver[Matrix[Double], Vector[Double]] =
      new DistBCFW(crfModel, roundLimit = 50,
        useCocoaPlus = false, debug = true,
        debugMultiplier = 1, debugOutPath = debugPath,
        samplePerRound = 1.0, doWeightedAveraging = false)*/

    val solver: DistributedSolver[Matrix[Double], Vector[Double]] =
      new DistSSGD(crfModel, roundLimit = 10,
        debug = true,
        debugMultiplier = 1, debugOutPath = debugPath,
        samplePerRound = 1.0, doWeightedAveraging = false)

    println("Running Distributed BCFW with CoCoA. Loaded data with %d rows, pattern=%dx%d, label=%dx1"
      .format(trainData.size, trainData(0).pattern.rows, trainData(0).pattern.cols, trainData(0).label.size))

    val classifier = new SSVMClassifier(crfModel)

    classifier.train(trainDataRDD, testDataRDD, solver)

    val x = trainData(0).pattern
    val y = classifier.predict(x)

    classifier.saveWeights("chain-weights.csv")

    classifier.loadWeights("chain-weights.csv")

  }

  def dbcfwExpt(): Unit = {

    val dataDir = "data/generated"
    val appname = "chain"
    val numStates = 26

    /**
     * Set and configure Spark
     */
    val conf = new SparkConf().setAppName(appname)
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
      sc.parallelize(testData)

    val trainDataRDD =
      sc.parallelize(trainData)

    /**
     * Train model
     */
    val crfModel = new LinearChainCRF(numStates, disablePairwise = false, useBPDecoding = false)

    val classifier = new SSVMClassifier(crfModel)

    val numPartsList = List(2, 4, 8, 16)
    val sampleFracList = List(1.0)

    for (numParts <- numPartsList) {

      for (sampleFrac <- sampleFracList) {

        println("Beginning Experiment with: nParts = %d, samplePerRound = %f".format(numParts, sampleFrac))

        val debugPath = "%d-chain-parts_%d-frac_%f.csv".format(System.currentTimeMillis() / 1000, numParts, sampleFrac)

        val thisTraining = trainDataRDD.repartition(numParts)
        val thisTest = testDataRDD.repartition(numParts)

        val solver: DistributedSolver[Matrix[Double], Vector[Double]] =
          new DistBCFW(crfModel, roundLimit = 60,
            useCocoaPlus = false, debug = true,
            debugMultiplier = 2, debugOutPath = debugPath,
            samplePerRound = sampleFrac, doWeightedAveraging = false)

        classifier.train(thisTraining, thisTest, solver)

        classifier.saveWeights("%d-chain-parts_%d-frac_%f-weights.csv".format(System.currentTimeMillis() / 1000, numParts, sampleFrac))

      }

    }

  }

  def dbcfw_speedup(args: Array[String]): Unit = {

    val dataDir = "data/generated"
    val appname = "chain"
    val numStates = 26

    val prefix = args(0)
    val sampleFrac = args(1).toDouble
    val numParts = args(2).toInt

    println("prefix = %s\tsampleFrac = %f\tnumParts = %d".format(prefix, sampleFrac, numParts))

    /**
     * Set and configure Spark
     */
    val conf = new SparkConf().setAppName(appname)
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
      sc.parallelize(testData, numParts)

    val trainDataRDD =
      sc.parallelize(trainData, numParts)

    /**
     * Train model
     */
    val crfModel = new LinearChainCRF(numStates, disablePairwise = false, useBPDecoding = false)

    val classifier = new SSVMClassifier(crfModel)

    println("Beginning Experiment with: nParts = %d, samplePerRound = %f".format(numParts, sampleFrac))

    val debugPath = "%d-chain-%s-parts_%d-frac_%f.csv"
      .format(System.currentTimeMillis() / 1000, prefix, numParts, sampleFrac)

    val thisTraining = trainDataRDD.repartition(numParts)
    val thisTest = testDataRDD.repartition(numParts)

    val solver: DistributedSolver[Matrix[Double], Vector[Double]] =
      new DistBCFW(crfModel, roundLimit = 100,
        useCocoaPlus = false, debug = true,
        debugMultiplier = 2, debugOutPath = debugPath,
        samplePerRound = sampleFrac, doWeightedAveraging = false)

    classifier.train(thisTraining, thisTest, solver)

    classifier.saveWeights("%d-chain-%s-parts_%d-frac_%f-weights.csv"
      .format(System.currentTimeMillis() / 1000, prefix, numParts, sampleFrac))

  }

  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("conf/log4j.properties")

    System.setProperty("spark.akka.frameSize", "512")

    // chainDBCFWSolver()

    // chainBCFW()

    dbcfw_speedup(args)
  }

}