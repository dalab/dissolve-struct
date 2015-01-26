/**
 *
 */
package ch.ethz.dalab.dissolve.test.bcfw

import org.scalatest.FunSpec
import org.scalatest.GivenWhenThen
import org.scalatest.BeforeAndAfter
import breeze.linalg._
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dal.dissolve.examples.chain.ChainDemo.{ loadData, featureFn, lossFn, oracleFn, predictFn }
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import org.apache.log4j.PropertyConfigurator
import org.apache.log4j.Level
import org.apache.log4j.Logger

/**
 * @author tribhu
 *
 */
class ChainRegression extends FunSpec with GivenWhenThen {

  def compareResults(baselineResults: String, currentResults: String): Unit = {

    def getLines(filePath: String): List[String] = scala.io.Source.fromFile(filePath).getLines()
      .toList
      .filter { line => line.length() > 5 }
      .filter { line => line(0) != '#' }

    def convertDataToMap(header: String, resultLine: String): Map[String, Double] = {
      val headerElements = header.split(",")
      val resultElements = resultLine.split(",").map { res => res.toDouble }

      assert(headerElements.length == resultElements.length)

      headerElements.zip(resultElements).toMap
    }

    val COLS_TO_CHECK = List(
      "primal",
      "dual",
      "gap",
      "train_error",
      "test_error")

    val baselineResultsLines = getLines(baselineResults)
    val currentResultsLines = getLines(currentResults)

    // Separate the Header of the CSV file and the data
    val baselineResultsHeader = baselineResultsLines(0)
    val currentResultsHeader = currentResultsLines(0)
    val baselineData = baselineResultsLines.tail
    val currentData = currentResultsLines.tail

    assert(baselineData.length == currentData.length, "Number of Baseline and Current Results do not match")

    // Convert each data result line to a map
    val bs = baselineData.map(d => convertDataToMap(baselineResultsHeader, d))
    val cur = currentData.map(d => convertDataToMap(currentResultsHeader, d))

    bs.zip(cur).map {
      case (baseline, current) =>
        for (col <- COLS_TO_CHECK) {
          assert(baseline("round") == current("round"))
          assert(baseline(col) == current(col), "Baseline(%s) = %f, Current(%s) = %f in Round %d"
            .format(col, baseline(col), col, current(col), baseline("round").toInt))
        }
    }

  }

  describe("Dissolve-Chain-Regression") {

    /**
     *  Initialize Spark Context
     */
    PropertyConfigurator.configure("conf/log4j.properties")
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val conf = new SparkConf().setAppName("Dissolve-Chain-Regression").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    /**
     * Load Data
     */
    val PERC_TRAIN: Double = 0.05

    val trainDataUnord: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val testDataUnord: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    println("Loaded data with %d rows, pattern=%dx%d, label=%dx1".format(trainDataUnord.size, trainDataUnord(0).pattern.rows, trainDataUnord(0).pattern.cols, trainDataUnord(0).label.size))

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    val train_data: Array[LabeledObject[Matrix[Double], Vector[Double]]] = trainDataUnord(List.fromArray(perm).slice(0, (PERC_TRAIN * trainDataUnord.size).toInt)).toArray

    /**
     *  Start with a default setting of Solver Options
     */
    val solverOptions: SolverOptions[Matrix[Double], Vector[Double]] = new SolverOptions()
    solverOptions.numPasses = 5
    solverOptions.debug = true
    solverOptions.lambda = 0.01
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = true
    solverOptions.testDataRDD = Some(sc.parallelize(testDataUnord.toArray, solverOptions.NUM_PART))
    solverOptions.debugInfoPath = "debug/regression-chain.csv".format(System.currentTimeMillis())
    solverOptions.enableManualPartitionSize = true

    solverOptions.doWeightedAveraging = false
    solverOptions.enableOracleCache = false
    solverOptions.oracleCacheSize = 10
    solverOptions.NUM_PART = 1
    solverOptions.sample = "frac"
    solverOptions.sampleFrac = 1.0

    /**
     * Start the tests
     */

    /**
     * ---------------------------------------------------------------------------------------------------------
     */
    it("""
      |should be consistent with single-node BCFW results without wAvg nor cache:
      |wAvg = false,
      |NUM_PART = 1,
      |cache = disabled,
      |sampleFrac = 1.0
      |compareWith = dissolve-base.csv""".stripMargin) {

      Given("Base config")

      Given("Baseline")
      val baselinePath = "data/regression-tests/dissolve-base.csv"

      val trainDataRDD = sc.parallelize(train_data, solverOptions.NUM_PART)

      When("Model is trained")
      val trainer: StructSVMWithDBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithDBCFW[Matrix[Double], Vector[Double]](
        trainDataRDD,
        featureFn,
        lossFn,
        oracleFn,
        predictFn,
        solverOptions)
      val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

      Then("Primal values and Errors should match baseline data")
      compareResults(baselinePath, "debug/regression-chain.csv")

    }

    /**
     * ---------------------------------------------------------------------------------------------------------
     */
    it("""
      |should be consistent with single-node BCFW results with wAvg:
      |wAvg = true,
      |NUM_PART = 1,
      |cache = disabled,
      |sampleFrac = 1.0
      |compareWith = dissolve-base-wavg.csv""".stripMargin) {

      Given("Base config + wavg")
      solverOptions.doWeightedAveraging = true

      Given("Baseline")
      val baselinePath = "data/regression-tests/dissolve-base-wavg.csv"

      val trainDataRDD = sc.parallelize(train_data, solverOptions.NUM_PART)

      When("Model is trained")
      val trainer: StructSVMWithDBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithDBCFW[Matrix[Double], Vector[Double]](
        trainDataRDD,
        featureFn,
        lossFn,
        oracleFn,
        predictFn,
        solverOptions)
      val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

      Then("Primal values and Errors should match baseline data")
      compareResults(baselinePath, "debug/regression-chain.csv")

      Then("Revert to Base config")
      solverOptions.doWeightedAveraging = false
    }

    /**
     * ---------------------------------------------------------------------------------------------------------
     */
    it("""
      |should be consistent with single-node BCFW results with cache:
      |wAvg = false,
      |NUM_PART = 1,
      |cache = size(10),
      |sampleFrac = 1.0
      |compareWith = dissolve-base-cache.csv""".stripMargin) {

      Given("Base config + cache")
      solverOptions.enableOracleCache = true

      Given("Baseline")
      val baselinePath = "data/regression-tests/dissolve-base-cache.csv"

      val trainDataRDD = sc.parallelize(train_data, solverOptions.NUM_PART)

      When("Model is trained")
      val trainer: StructSVMWithDBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithDBCFW[Matrix[Double], Vector[Double]](
        trainDataRDD,
        featureFn,
        lossFn,
        oracleFn,
        predictFn,
        solverOptions)
      val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

      Then("Primal values and Errors should match baseline data")
      compareResults(baselinePath, "debug/regression-chain.csv")

      Then("Revert to Base config")
      solverOptions.enableOracleCache = false
    }

    /**
     * ---------------------------------------------------------------------------------------------------------
     */
    it("""
      |should be consistent with previous results over 5 passes when:
      |wAvg = false,
      |NUM_PART = 1,
      |cache = disabled,
      |sampleFrac = 0.5
      |compareWith = dissolve-frac.csv""".stripMargin) {

      Given("Base config + frac")
      solverOptions.sampleFrac = 0.5

      Given("Baseline")
      val baselinePath = "data/regression-tests/dissolve-frac.csv"

      val trainDataRDD = sc.parallelize(train_data, solverOptions.NUM_PART)

      When("Model is trained")
      val trainer: StructSVMWithDBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithDBCFW[Matrix[Double], Vector[Double]](
        trainDataRDD,
        featureFn,
        lossFn,
        oracleFn,
        predictFn,
        solverOptions)
      val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

      Then("Primal values and Errors should match baseline data")
      compareResults(baselinePath, "debug/regression-chain.csv")

      Then("Revert to Base config")
      solverOptions.sampleFrac = 1.0
    }

    /**
     * ---------------------------------------------------------------------------------------------------------
     */
    it("""
      |should be consistent with previous results over 5 passes when:
      |wAvg = false,
      |NUM_PART = 4,
      |cache = disabled,
      |sampleFrac = 0.5
      |compareWith = dissolve-frac-parts.csv""".stripMargin) {

      Given("Base config + frac + partitions")
      solverOptions.sampleFrac = 0.5
      solverOptions.NUM_PART = 4

      Given("Baseline")
      val baselinePath = "data/regression-tests/dissolve-frac-parts.csv"

      val trainDataRDD = sc.parallelize(train_data, solverOptions.NUM_PART)

      When("Model is trained")
      val trainer: StructSVMWithDBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithDBCFW[Matrix[Double], Vector[Double]](
        trainDataRDD,
        featureFn,
        lossFn,
        oracleFn,
        predictFn,
        solverOptions)
      val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

      Then("Primal values and Errors should match baseline data")
      compareResults(baselinePath, "debug/regression-chain.csv")

      Then("Revert to Base config")
      solverOptions.sampleFrac = 1.0
      solverOptions.NUM_PART = 1
    }

    /**
     * ---------------------------------------------------------------------------------------------------------
     */
    it("""
      |should be consistent with previous results over 5 passes when:
      |wAvg = true,
      |NUM_PART = 4,
      |cache = disabled,
      |sampleFrac = 0.5
      |compareWith = dissolve-frac-parts-wavg.csv""".stripMargin) {

      Given("Base config + frac + partitions + wAvg")
      solverOptions.sampleFrac = 0.5
      solverOptions.NUM_PART = 4
      solverOptions.doWeightedAveraging = true

      Given("Baseline")
      val baselinePath = "data/regression-tests/dissolve-frac-parts-wavg.csv"

      val trainDataRDD = sc.parallelize(train_data, solverOptions.NUM_PART)

      When("Model is trained")
      val trainer: StructSVMWithDBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithDBCFW[Matrix[Double], Vector[Double]](
        trainDataRDD,
        featureFn,
        lossFn,
        oracleFn,
        predictFn,
        solverOptions)
      val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

      Then("Primal values and Errors should match baseline data")
      compareResults(baselinePath, "debug/regression-chain.csv")

      Then("Revert to Base config")
      solverOptions.sampleFrac = 1.0
      solverOptions.NUM_PART = 1
      solverOptions.doWeightedAveraging = false
    }

    /**
     * ---------------------------------------------------------------------------------------------------------
     */
    it("""
      |should be consistent with previous results over 5 passes when:
      |wAvg = true,
      |NUM_PART = 4,
      |cache = size(10),
      |sampleFrac = 0.5
      |compareWith = dissolve-frac-parts-wavg-cache.csv""".stripMargin) {

      Given("Base config + frac + partitions + wAvg + cache")
      solverOptions.sampleFrac = 0.5
      solverOptions.NUM_PART = 4
      solverOptions.doWeightedAveraging = true
      solverOptions.enableOracleCache = true

      Given("Baseline")
      val baselinePath = "data/regression-tests/dissolve-frac-parts-wavg-cache.csv"

      val trainDataRDD = sc.parallelize(train_data, solverOptions.NUM_PART)

      When("Model is trained")
      val trainer: StructSVMWithDBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithDBCFW[Matrix[Double], Vector[Double]](
        trainDataRDD,
        featureFn,
        lossFn,
        oracleFn,
        predictFn,
        solverOptions)
      val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

      Then("Primal values and Errors should match baseline data")
      compareResults(baselinePath, "debug/regression-chain.csv")

      Then("Revert to Base config")
      solverOptions.sampleFrac = 1.0
      solverOptions.NUM_PART = 1
      solverOptions.doWeightedAveraging = false
      solverOptions.enableOracleCache = false
    }
  }
}

object ChainRegression {
  def main(args: Array[String]): Unit = {
    (new ChainRegression).execute()
  }
}