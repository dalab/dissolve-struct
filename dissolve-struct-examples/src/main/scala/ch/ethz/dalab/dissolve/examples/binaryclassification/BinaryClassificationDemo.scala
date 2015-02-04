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
import org.apache.log4j.PropertyConfigurator
import org.apache.spark.mllib.linalg.Vectors
import breeze.linalg._

/**
 * Runs the binary 
 *
 * Created by tribhu on 12/11/14.
 */
object BinaryClassificationDemo {

  /**
   * Training Implementation
   */
  def dissolveTrain(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint]) {

    val solverOptions: SolverOptions[Vector[Double], Double] = new SolverOptions()

    solverOptions.numRounds = 20 // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = true
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debug = false

    solverOptions.sampleWithReplacement = false

    solverOptions.enableManualPartitionSize = true
    solverOptions.NUM_PART = 2

    solverOptions.sample = "frac"
    solverOptions.sampleFrac = 0.5

    solverOptions.enableOracleCache = false

    solverOptions.debugInfoPath = "../debug/debugInfo-%d.csv".format(System.currentTimeMillis())

    val trainDataConverted: RDD[LabeledObject[Vector[Double], Double]] =
      trainData.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], Double](x.label, Vector(x.features.toArray))
      }

    val testDataConverted: RDD[LabeledObject[Vector[Double], Double]] =
      testData.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], Double](x.label, Vector(x.features.toArray))
      }

    solverOptions.testDataRDD = Some(testDataConverted)
    val model = BinarySVMWithDBCFW.train(trainData, solverOptions)

    // Training Errors
    val trueTrainingPredictions =
      trainDataConverted.map {
        case x: LabeledObject[Vector[Double], Double] =>
          val prediction = model.predict(x.pattern)
          if (prediction == x.label)
            1
          else
            0
      }.fold(0)((acc, ele) => acc + ele)

    println("Accuracy on training set = %d/%d = %.4f".format(trueTrainingPredictions,
      trainDataConverted.count(),
      (trueTrainingPredictions.toDouble / trainDataConverted.count().toDouble) * 100))

    // Test Errors
    val trueTestPredictions =
      testDataConverted.map {
        case x: LabeledObject[Vector[Double], Double] =>
          val prediction = model.predict(x.pattern)
          if (prediction == x.label)
            1
          else
            0
      }.fold(0)((acc, ele) => acc + ele)

    println("Accuracy on test set = %d/%d = %.4f".format(trueTestPredictions,
      testDataConverted.count(),
      (trueTestPredictions.toDouble / testDataConverted.count().toDouble) * 100))
  }

  /**
   * MLLib's SVMWithSGD implementation
   */
  def mllibTrain(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint]) {
    println("running MLlib's standard gradient descent solver")
    
    // labels are assumed to be 0,1 for MLlib
    val trainDataConverted: RDD[LabeledPoint] =
      trainData.map {
        case x: LabeledPoint =>
          new LabeledPoint(if (x.label > 0) 1 else 0, x.features)
      }

    val testDataConverted: RDD[LabeledPoint] =
      testData.map {
        case x: LabeledPoint =>
          new LabeledPoint(if (x.label > 0) 1 else 0, x.features)
      }

    // Run training algorithm to build the model
    val model = SVMWithSGD.train(trainDataConverted, numIterations = 200, stepSize = 1.0, regParam = 0.01)
    
    // report accuracy on train and test
    val trainAcc = testDataConverted.map { point =>
      val score = model.predict(point.features)
      score == point.label
    }.collect().toList.count(_ == true).toDouble / testDataConverted.count().toDouble

    val testAcc = testDataConverted.map { point =>
      val score = model.predict(point.features)
      score == point.label
    }.collect().toList.count(_ == true).toDouble / testDataConverted.count().toDouble

    println("MLlib Training accuracy = " + trainAcc)
    println("MLlib Test accuracy = " + testAcc)

  }

  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("conf/log4j.properties")
    
    val dataDir = "../data/generated"
      
    // Dataset: Adult (http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a)
    val datasetFilename = "a1a"

    val conf = new SparkConf().setAppName("BinaryClassificationDemo").setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    // load dataset
    val data = MLUtils.loadLibSVMFile(sc, dataDir+"/"+datasetFilename)
    
    // Split data into training and test set
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1L)
    val trainingData = splits(0)
    val testData = splits(1)

    // Fix seed for reproducibility
    util.Random.setSeed(1)
    
    
    mllibTrain(trainingData, testData)

    dissolveTrain(trainingData, testData)
  }

}
