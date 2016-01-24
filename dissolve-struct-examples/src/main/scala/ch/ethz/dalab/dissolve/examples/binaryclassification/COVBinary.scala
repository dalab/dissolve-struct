package ch.ethz.dalab.dissolve.examples.binaryclassification

import java.io.File
import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.optimization.BinaryClassifier
import ch.ethz.dalab.dissolve.optimization.DistBCFW
import ch.ethz.dalab.dissolve.optimization.LocalBCFW

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
    val appname = "cov"
    val covPath = "data/generated/covtype.libsvm.binary.scale"
    val debugPath = "cov-%d.csv".format(System.currentTimeMillis() / 1000)
    val numPartitions = 6

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
    // val splits = data.randomSplit(Array(0.1, 0.1, 0.8), seed = 1L)
    val training = splits(0)
    val test = splits(1)

    val classifier = new BinaryClassifier(invFreqLoss = true)
    val model = classifier.getModel()
    val solver = new DistBCFW(model, roundLimit = 50,
      useCocoaPlus = false, debug = true,
      debugMultiplier = 2, debugOutPath = debugPath,
      samplePerRound = 1.0, doWeightedAveraging = false)

    classifier.train(training, test, solver)

  }

  def dbcfwExpt(args: Array[String]) {
    val appname = "cov"
    val covPath = "data/generated/covtype.libsvm.binary.scale"

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
    // val splits = data.randomSplit(Array(0.1, 0.1, 0.8), seed = 1L)
    val training = splits(0)
    val test = splits(1)

    val numPartsList = List(2, 4, 8, 16)
    val sampleFracList = List(1.0)

    for (numParts <- numPartsList) {

      for (sampleFrac <- sampleFracList) {

        println("Beginning Experiment with: nParts = %d, samplePerRound = %f".format(numParts, sampleFrac))

        val thisTraining = training.repartition(numParts)
        val thisTest = test.repartition(numParts)

        val debugPath = "%d-cov-parts_%d-frac_%f.csv".format(System.currentTimeMillis() / 1000, numParts, sampleFrac)

        val classifier = new BinaryClassifier(invFreqLoss = true)
        val model = classifier.getModel()
        val solver = new DistBCFW(model, roundLimit = 200,
          useCocoaPlus = false, debug = true,
          debugMultiplier = 2, debugOutPath = debugPath,
          samplePerRound = sampleFrac, doWeightedAveraging = false)

        classifier.train(training, test, solver)

        classifier.saveWeights("%d-cov-parts_%d-frac_%f-weights.csv".format(System.currentTimeMillis() / 1000, numParts, sampleFrac))
      }
    }

  }

  def main(args: Array[String]): Unit = {

    PropertyConfigurator.configure("conf/log4j.properties")

    System.setProperty("spark.akka.frameSize", "512")

    dbcfwExpt(args)
  }

}