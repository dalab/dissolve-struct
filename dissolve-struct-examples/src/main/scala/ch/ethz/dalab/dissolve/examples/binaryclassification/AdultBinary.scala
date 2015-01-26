package ch.ethz.dalab.dissolve.examples.binaryclassification

import ch.ethz.dalab.dissolve.utils.DissolveUtils
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
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

    // Fix seed for reproducibility
    util.Random.setSeed(1)

    val conf = new SparkConf().setAppName("Adult-example").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    val solverOptions: SolverOptions[Vector[Double], Double] = new SolverOptions()

    solverOptions.numPasses = 20 // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = true
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = false

    solverOptions.sampleWithReplacement = false

    solverOptions.enableManualPartitionSize = true
    solverOptions.NUM_PART = 1

    solverOptions.sample = "frac"
    solverOptions.sampleFrac = 0.5

    solverOptions.enableOracleCache = false

    solverOptions.debugInfoPath = "../debug/debugInfo-a1a-%d.csv".format(System.currentTimeMillis())


    val data = MLUtils.loadLibSVMFile(sc, a1aPath)

    // Split data into training and test set
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1L)
    val training = splits(0)
    val test = splits(1)

    val objectifiedTraining: RDD[LabeledObject[Vector[Double], Double]] =
      training.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], Double](x.label, Vector(x.features.toArray)) // Is the asInstanceOf required?
      }

    val objectifiedTest: RDD[LabeledObject[Vector[Double], Double]] =
      test.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], Double](x.label, Vector(x.features.toArray)) // Is the asInstanceOf required?
      }

    solverOptions.testDataRDD = Some(objectifiedTest)
    val model = BinarySVMWithDBCFW.train(training, solverOptions)

    // Training Errors
    val trueTrainingPredictions =
      objectifiedTraining.map {
        case x: LabeledObject[Vector[Double], Double] =>
          val prediction = model.predict(x.pattern)
          if (prediction == x.label)
            1
          else
            0
      }.fold(0)((acc, ele) => acc + ele)

    println("Accuracy on Training set = %d/%d = %.4f".format(trueTrainingPredictions,
      objectifiedTraining.count(),
      (trueTrainingPredictions.toDouble / objectifiedTraining.count().toDouble) * 100))

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
