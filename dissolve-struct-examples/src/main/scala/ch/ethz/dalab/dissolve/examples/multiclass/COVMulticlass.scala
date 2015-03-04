package ch.ethz.dalab.dissolve.examples.multiclass

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.classification.MultiClassLabel
import ch.ethz.dalab.dissolve.classification.MultiClassSVMWithDBCFW
import ch.ethz.dalab.dissolve.examples.utils.ExampleUtils
import ch.ethz.dalab.dissolve.optimization.GapThresholdCriterion
import ch.ethz.dalab.dissolve.optimization.RoundLimitCriterion
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.optimization.TimeLimitCriterion
import ch.ethz.dalab.dissolve.regression.LabeledObject

object COVMulticlass {

  def dissoveCovMulti(options: Map[String, String]) {

    /**
     * Load all options
     */
    val prefix: String = "cov-multi"
    val appName: String = options.getOrElse("appname", ExampleUtils
      .generateExperimentName(prefix = List(prefix, "%d".format(System.currentTimeMillis() / 1000))))

    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")

    val runLocally: Boolean = options.getOrElse("local", "false").toBoolean

    val solverOptions: SolverOptions[Vector[Double], MultiClassLabel] = new SolverOptions()
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean
    solverOptions.lambda = options.getOrElse("lambda", "0.01").toDouble
    solverOptions.doWeightedAveraging = options.getOrElse("wavg", "false").toBoolean
    solverOptions.doLineSearch = options.getOrElse("linesearch", "true").toBoolean

    solverOptions.sample = options.getOrElse("sample", "frac")
    solverOptions.sampleFrac = options.getOrElse("samplefrac", "0.5").toDouble
    solverOptions.sampleWithReplacement = options.getOrElse("samplewithreplacement", "false").toBoolean

    solverOptions.enableManualPartitionSize = options.getOrElse("manualrddpart", "false").toBoolean
    solverOptions.NUM_PART = options.getOrElse("numpart", "2").toInt

    solverOptions.enableOracleCache = options.getOrElse("enableoracle", "false").toBoolean
    solverOptions.oracleCacheSize = options.getOrElse("oraclesize", "5").toInt

    solverOptions.debugMultiplier = options.getOrElse("debugmultiplier", "5").toInt

    solverOptions.checkpointFreq = options.getOrElse("checkpointfreq", "50").toInt

    solverOptions.sparse = options.getOrElse("sparse", "false").toBoolean

    options.getOrElse("stoppingcriterion", "round") match {
      case "round" =>
        solverOptions.stoppingCriterion = RoundLimitCriterion
        solverOptions.roundLimit = options.getOrElse("roundlimit", "25").toInt
      case "gap" =>
        solverOptions.stoppingCriterion = GapThresholdCriterion
        solverOptions.gapThreshold = options.getOrElse("gapthreshold", "0.1").toDouble
        solverOptions.gapCheck = options.getOrElse("gapcheck", "10").toInt
      case "time" =>
        solverOptions.stoppingCriterion = TimeLimitCriterion
        solverOptions.timeLimit = options.getOrElse("timelimit", "300").toInt
      case _ =>
        println("Unrecognized Stopping Criterion. Moving to default criterion.")
    }

    solverOptions.debugInfoPath = options.getOrElse("debugpath", debugDir + "/%s.csv".format(appName))

    val defaultCovPath = dataDir + "/covtype.scale.head"
    val covPath = options.getOrElse("traindata", defaultCovPath)

    /**
     * Some local overrides
     */
    if (runLocally) {
      solverOptions.sampleFrac = 0.2
      solverOptions.enableOracleCache = false
      solverOptions.oracleCacheSize = 10
      solverOptions.enableManualPartitionSize = true
      solverOptions.NUM_PART = 1
      solverOptions.doWeightedAveraging = false

      solverOptions.stoppingCriterion = RoundLimitCriterion
      solverOptions.roundLimit = 5

      solverOptions.debug = true
      solverOptions.debugMultiplier = 1
    }

    println(solverOptions.toString())

    // Fix seed for reproducibility
    util.Random.setSeed(1)

    val conf =
      if (runLocally)
        new SparkConf().setAppName(appName).setMaster("local")
      else
        new SparkConf().setAppName(appName)

    val sc = new SparkContext(conf)
    sc.setCheckpointDir(dataDir + "/checkpoint-files")

    // Needs labels \in [0, numClasses)
    val data: RDD[LabeledPoint] = MLUtils
      .loadLibSVMFile(sc, covPath)
      .map {
        case x: LabeledPoint =>
          val label = x.label - 1
          LabeledPoint(label, x.features)
      }

    val minlabel = data.map(_.label).min()
    val maxlabel = data.map(_.label).max()
    println("min = %f, max = %f".format(minlabel, maxlabel))

    // Split data into training and test set
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1L)
    val training = splits(0)
    val test = splits(1)

    val numClasses = 7

    val objectifiedTest: RDD[LabeledObject[Vector[Double], MultiClassLabel]] =
      test.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], MultiClassLabel](MultiClassLabel(x.label, numClasses),
            Vector(x.features.toArray))
      }

    solverOptions.testDataRDD = Some(objectifiedTest)
    val model = MultiClassSVMWithDBCFW.train(data, numClasses, solverOptions)

    // Test Errors
    val trueTestPredictions =
      objectifiedTest.map {
        case x: LabeledObject[Vector[Double], MultiClassLabel] =>
          val prediction = model.predict(x.pattern)
          if (prediction == x.label)
            1
          else
            0
      }.fold(0)((acc, ele) => acc + ele)

    println("Accuracy on Test set = %d/%d = %.4f".format(trueTestPredictions,
      objectifiedTest.count(),
      (trueTestPredictions.toDouble / objectifiedTest.count().toDouble) * 100))

  }

  def main(args: Array[String]): Unit = {

    PropertyConfigurator.configure("conf/log4j.properties")

    val options: Map[String, String] = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt)    => (opt -> "true")
        case _             => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    System.setProperty("spark.akka.frameSize", "512")
    println(options)

    dissoveCovMulti(options)
  }

}