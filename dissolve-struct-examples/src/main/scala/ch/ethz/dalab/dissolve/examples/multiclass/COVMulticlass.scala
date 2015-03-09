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
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.utils.cli.CLAParser

object COVMulticlass {

  def dissoveCovMulti(args: Array[String]) {

    /**
     * Load all options
     */
    val (solverOptions, kwargs) = CLAParser.argsToOptions[Vector[Double], MultiClassLabel](args)
    val covPath = kwargs.getOrElse("input_path", "../data/generated/covtype.scale")
    val appname = kwargs.getOrElse("appname", "cov_multi")
    val debugPath = kwargs.getOrElse("debug_file", "cov_multi-%d.csv".format(System.currentTimeMillis() / 1000))
    solverOptions.debugInfoPath = debugPath
    
    println(covPath)
    println(kwargs)

    // Fix seed for reproducibility
    util.Random.setSeed(1)

    val conf = new SparkConf().setAppName(appname)

    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

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
    dissoveCovMulti(args)
  }

}