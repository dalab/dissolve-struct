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
import ch.ethz.dalab.dissolve.optimization.LocalBCFW
import ch.ethz.dalab.dissolve.optimization.LocalSSGD

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

    val model: StructSVMModel[Matrix[Double], Vector[Double]] = solver.train(train_data, test_data)

  }

  def chainDBCFWSolver(): Unit = {

    val dataDir = "../data/generated"
    val appname = "chain"
    val debugPath = "chain-%d.csv".format(System.currentTimeMillis() / 1000)
    val numPartitions = 1
    val numStates = 26

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
      Some(sc.parallelize(testData, numPartitions))

    val trainDataRDD =
      sc.parallelize(trainData, numPartitions)

    /**
     * Train model
     */
    val crfModel = new LinearChainCRF(numStates, disablePairwise = false, useBPDecoding = false)

    val solver: DistributedSolver[Matrix[Double], Vector[Double]] =
      new DistBCFW(crfModel, roundLimit = 10, debug = true, debugMultiplier = 2, debugOutPath = debugPath, samplePerRound = 1.0)

    println("Running Distributed BCFW with CoCoA. Loaded data with %d rows, pattern=%dx%d, label=%dx1"
      .format(trainData.size, trainData(0).pattern.rows, trainData(0).pattern.cols, trainData(0).label.size))

    val model: StructSVMModel[Matrix[Double], Vector[Double]] = solver.train(trainDataRDD, testDataRDD)

  }

  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("conf/log4j.properties")

    System.setProperty("spark.akka.frameSize", "512")

    chainDBCFWSolver()

    // chainBCFW()
  }

}