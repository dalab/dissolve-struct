package ch.ethz.dalab.dissolve.examples.binaryclassification

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import breeze.linalg.SparseVector
import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.classification.BinarySVMWithDBCFW
import ch.ethz.dalab.dissolve.examples.utils.ExampleUtils
import ch.ethz.dalab.dissolve.optimization.GapThresholdCriterion
import ch.ethz.dalab.dissolve.optimization.RoundLimitCriterion
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.optimization.TimeLimitCriterion
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.utils.cli.CLAParser

object RCV1Binary {

  def dbcfwRcv1(args: Array[String]) {
    /**
     * Load all options
     */
    val (solverOptions, kwargs) = CLAParser.argsToOptions[Vector[Double], Double](args)
    val rcv1Path = kwargs.getOrElse("input_path", "../data/generated/rcv1_train.binary")
    val appname = kwargs.getOrElse("appname", "rcv1_binary")
    val debugPath = kwargs.getOrElse("debug_file", "rcv1_binary-%d.csv".format(System.currentTimeMillis() / 1000))
    solverOptions.debugInfoPath = debugPath
    solverOptions.sparse = true
    solverOptions.classWeights = true
    
    solverOptions.stoppingCriterion = GapThresholdCriterion
    solverOptions.gapThreshold = 1e-3
    solverOptions.gapCheck = 10 // Checks for gap every gapCheck rounds
    
    solverOptions.debug  = true
    solverOptions.debugMultiplier = 2
    
    println(rcv1Path)
    println(kwargs)

    // Fix seed for reproducibility
    util.Random.setSeed(1)

    println(solverOptions.toString())

    val conf = new SparkConf().setAppName(appname).setMaster("local")
    conf.set("spark.driver.memory", "12g")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")
    
    System.setProperty("hadoop.home.dir", "C:/winutil/")

    // Labels needs to be in a +1/-1 format
    val data = MLUtils.loadLibSVMFile(sc, rcv1Path)

    // Split data into training and test set
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 1L)
    val training = splits(0)
    val test = splits(1)

    val objectifiedTest: RDD[LabeledObject[Vector[Double], Double]] =
      test.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], Double](x.label, SparseVector(x.features.toArray)) // Is the asInstanceOf required?
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

    dbcfwRcv1(args)

  }

}