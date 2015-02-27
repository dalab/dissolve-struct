package ch.ethz.dalab.dissolve.examples.binaryclassification

import ch.ethz.dalab.dissolve.utils.DissolveUtils
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.classification.BinarySVMWithDBCFW
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD
import breeze.linalg._
import breeze.numerics.abs
import org.apache.log4j.PropertyConfigurator
import ch.ethz.dalab.dissolve.examples.utils.ExampleUtils

object COVBinary {

  /**
   * MLLib's classifier
   */
  def mllibCov() {
    val conf = new SparkConf().setAppName("Adult-example").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    val data = MLUtils.loadLibSVMFile(sc, "../data/generated/covtype.libsvm.binary.scale.head.mllib")

    // Split data into training and test set
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1L)
    val training = splits(0)
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 1000
    val model = SVMWithSGD.train(training, numIterations)

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

  /**
   * DBCFW classifier
   */
  def dbcfwCov(options: Map[String, String]) {
    /**
     * Load all options
     */
    val prefix: String = "cov"
    val appName: String = options.getOrElse("appname", ExampleUtils
      .generateExperimentName(prefix = List(prefix, "%d".format(System.currentTimeMillis() / 1000))))

    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")

    val solverOptions: SolverOptions[Vector[Double], Double] = new SolverOptions()
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

    options.getOrElse("stoppingcriterion", "round") match {
      case "round" =>
        solverOptions.stoppingCriterion = solverOptions.RoundLimitCriterion
        solverOptions.roundLimit = options.getOrElse("roundlimit", "25").toInt
      case "gap" =>
        solverOptions.stoppingCriterion = solverOptions.GapThresholdCriterion
        solverOptions.gapThreshold = options.getOrElse("gapthreshold", "0.1").toDouble
        solverOptions.gapCheck = options.getOrElse("gapcheck", "10").toInt
      case "time" =>
        solverOptions.stoppingCriterion = solverOptions.TimeLimitCriterion
        solverOptions.timeLimit = options.getOrElse("timelimit", "300").toInt
      case _ =>
        println("Unrecognized Stopping Criterion. Moving to default criterion.")
    }

    solverOptions.debugInfoPath = options.getOrElse("debugpath", debugDir + "/%s.csv".format(appName))

    val defaultCovPath = dataDir + "/covtype.libsvm.binary.scale"
    val covPath = options.getOrElse("traindata", defaultCovPath)

    // Fix seed for reproducibility
    util.Random.setSeed(1)

    val conf = new SparkConf().setAppName("COV-example").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(dataDir + "/checkpoint-files")

    // Labels needs to be in a +1/-1 format
    val data = MLUtils
      .loadLibSVMFile(sc, covPath)
      .map {
        case x: LabeledPoint =>
          val label =
            if (x.label == 1)
              +1.00
            else
              -1.00
          LabeledPoint(label, x.features)
      }

    // Split data into training and test set
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1L)
    val training = splits(0)
    val test = splits(1)

    val objectifiedTest: RDD[LabeledObject[Vector[Double], Double]] =
      test.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], Double](x.label, Vector(x.features.toArray)) // Is the asInstanceOf required?
      }

    solverOptions.testDataRDD = Some(objectifiedTest)
    val model = BinarySVMWithDBCFW.train(training, solverOptions)

    // Test Errors
    val trueTestPredictions =
      objectifiedTest.map {
        case x: LabeledObject[Vector[Double], Double] =>
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

    dbcfwCov(options)
  }

}