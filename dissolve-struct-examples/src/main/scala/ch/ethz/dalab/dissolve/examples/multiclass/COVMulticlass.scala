package ch.ethz.dalab.dissolve.examples.multiclass

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import breeze.linalg._

import ch.ethz.dalab.dissolve.optimization.DistBCFW
import ch.ethz.dalab.dissolve.optimization.MulticlassClassifier

object COVMulticlass {

  def dissoveCovMulti(args: Array[String]) {

    /**
     * Load all options
     */
    val appname = "covmul"
    val covPath = "../data/generated/covtype.scale"
    val debugPath = "covmul-%d.csv".format(System.currentTimeMillis() / 1000)
    val numPartitions = 6
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

    val classifier = new MulticlassClassifier(numClasses, invFreqLoss = true)
    val model = classifier.getModel()
    val solver = new DistBCFW(model, roundLimit = 50,
      useCocoaPlus = false, debug = true,
      debugMultiplier = 2, debugOutPath = debugPath,
      samplePerRound = 0.5, doWeightedAveraging = false)
    classifier.train(training, test, solver)

  }

  def main(args: Array[String]): Unit = {

    PropertyConfigurator.configure("conf/log4j.properties")

    System.setProperty("spark.akka.frameSize", "512")

    dissoveCovMulti(args)
  }

}