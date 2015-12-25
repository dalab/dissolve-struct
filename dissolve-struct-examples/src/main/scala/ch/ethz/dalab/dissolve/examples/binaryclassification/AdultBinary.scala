package ch.ethz.dalab.dissolve.examples.binaryclassification

import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.SVMWithSGD
import breeze.linalg._
import breeze.numerics.abs
import ch.ethz.dalab.dissolve.optimization.BinaryClassifier
import ch.ethz.dalab.dissolve.optimization.DistBCFW

/**
 *
 * Dataset: Adult (http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a)
 * Type: Binary
 *
 * Created by tribhu on 12/11/14.
 */
object AdultBinary {

  /**
   * DBCFW Implementation
   */
  def dbcfwAdult() {
    val a1aPath = "../data/generated/a1a"
    val appname = "a1a"
    val debugPath = "a1a-%d.csv".format(System.currentTimeMillis() / 1000)
    val numPartitions = 6

    // Fix seed for reproducibility
    util.Random.setSeed(1)

    val conf = new SparkConf().setAppName("Adult-example").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    val data = MLUtils.loadLibSVMFile(sc, a1aPath, numPartitions)

    // Split data into training and test set
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1L)
    val training = splits(0)
    val test = splits(1)

    val classifier = new BinaryClassifier(invFreqLoss = true)
    val model = classifier.getModel()
    val solver = new DistBCFW(model, roundLimit = 50,
      useCocoaPlus = false, debug = true,
      debugMultiplier = 1, debugOutPath = debugPath,
      samplePerRound = 1.0, doWeightedAveraging = false)
    classifier.train(training, test, solver)

  }

  /**
   * MLLib's SVMWithSGD implementation
   */
  def mllibAdult() {

    val conf = new SparkConf().setAppName("Adult-example").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    val data = MLUtils.loadLibSVMFile(sc, "../data/a1a_mllib.txt")

    // Split data into training and test set
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1L)
    val training = splits(0)
    val test = splits(1)

    // Run training algorithm to build the model
    val numIterations = 100
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

  def main(args: Array[String]): Unit = {
    // mllibAdult()

    dbcfwAdult()
  }

}
