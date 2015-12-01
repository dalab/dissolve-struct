package ch.ethz.dalab.dissolve.examples.binaryclassification

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.classification.BinarySVMWithDBCFW
import ch.ethz.dalab.dissolve.examples.utils.ExampleUtils
import ch.ethz.dalab.dissolve.optimization.GapThresholdCriterion
import ch.ethz.dalab.dissolve.optimization.RoundLimitCriterion
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.optimization.TimeLimitCriterion
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.utils.cli.CLAParser
import java.io.File

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
  def dbcfwCov(args: Array[String]) {
    /**
     * Load all options
     */
    val (solverOptions, kwargs) = CLAParser.argsToOptions[Vector[Double], Double](args)
    val covPath = kwargs.getOrElse("input_path", "../data/generated/covtype.libsvm.binary.scale")
    val appname = kwargs.getOrElse("appname", "cov_binary")
    val debugPath = kwargs.getOrElse("debug_file", "cov_binary-%d.csv".format(System.currentTimeMillis() / 1000))
    solverOptions.debugInfoPath = debugPath
    
    println(covPath)
    println(kwargs)
    
    println("Current directory:" + new File(".").getAbsolutePath)

    // Fix seed for reproducibility
    util.Random.setSeed(1)

    val conf = new SparkConf().setAppName(appname)
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

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

    System.setProperty("spark.akka.frameSize", "512")

    dbcfwCov(args)
  }

}