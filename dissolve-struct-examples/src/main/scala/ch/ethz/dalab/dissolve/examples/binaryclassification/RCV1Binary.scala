package ch.ethz.dalab.dissolve.examples.binaryclassification

import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import ch.ethz.dalab.dissolve.optimization.BinaryClassifier
import ch.ethz.dalab.dissolve.optimization.DistBCFW

object RCV1Binary {

  def dbcfwRcv1(args: Array[String]) {
    /**
     * Load all options
     */
    val appname = "rcv1"
    val rcv1Path = "../data/generated/rcv1_train.binary"
    val debugPath = "rcv1-%d.csv".format(System.currentTimeMillis() / 1000)
    val numPartitions = 6
    
    // Fix seed for reproducibility
    util.Random.setSeed(1)

    val conf = new SparkConf().setAppName(appname)
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    // Labels needs to be in a +1/-1 format
    val data = MLUtils.loadLibSVMFile(sc, rcv1Path)

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

  def main(args: Array[String]): Unit = {

    PropertyConfigurator.configure("conf/log4j.properties")

    System.setProperty("spark.akka.frameSize", "512")

    dbcfwRcv1(args)

  }

}