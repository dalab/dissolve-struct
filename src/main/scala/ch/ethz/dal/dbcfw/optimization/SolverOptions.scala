package ch.ethz.dal.dbcfw.optimization

import ch.ethz.dal.dbcfw.regression.LabeledObject
import breeze.linalg.Vector
import org.apache.spark.rdd.RDD
import java.io.File

class SolverOptions[X, Y] extends Serializable {
  var numPasses: Int = 50 // #Passes in case of BCFW, #Rounds in case of DBCFW
  var doWeightedAveraging: Boolean = true
  var timeBudget = Int.MaxValue

  var randSeed: Int = 42
  /**
   *  BCFW - "uniform", "perm" or "iter"
   *  DBCFW - "count", "frac"
   */
  var sample: String = "perm"
  var debugMultiplier: Int = 0
  var lambda: Double = 0.01 // FIXME This is 1/n in Matlab code
  
  var testData: Option[Seq[LabeledObject[X, Y]]] = Option.empty[Seq[LabeledObject[X, Y]]]
  var testDataRDD: Option[RDD[LabeledObject[X, Y]]] = Option.empty[RDD[LabeledObject[X, Y]]]
  
  var doLineSearch: Boolean = false

  // Cache params
  var enableOracleCache: Boolean = true
  var oracleCacheSize: Int = 10

  // DBCFW specific params
  var autoconfigure: Boolean = false
  var H: Int = 5 // Number of points to sample in each pass
  var sampleFrac: Double = 0.5
  var sampleWithReplacement: Boolean = false

  var enableManualPartitionSize: Boolean = false
  var NUM_PART: Int = 1 // Number of partitions of the RDD

  // For debugging/Testing purposes
  // Basic debugging flag
  var debug: Boolean = false
  // More verbose debugging flag
  var xldebug: Boolean = false
  // Write weights to CSV after each pass
  var debugWeights: Boolean = false
  // Dump loss through iterations
  var debugLoss: Boolean = true

  // Path to write the CSVs
  var debugInfoPath: String = new File(".").getCanonicalPath() + "/debugInfo-%d.csv".format(System.currentTimeMillis())

  override def toString(): String = {
    val sb: StringBuilder = new StringBuilder()

    sb ++= "# numPasses=%s\n".format(numPasses)
    sb ++= "# doWeightedAveraging=%s\n".format(doWeightedAveraging)

    sb ++= "# randSeed=%d\n".format(randSeed)

    sb ++= "# sample=%s\n".format(sample)
    sb ++= "# lambda=%f\n".format(lambda)
    sb ++= "# doLineSearch=%s\n".format(doLineSearch)

    sb ++= "# enableManualPartitionSize=%s\n".format(enableManualPartitionSize)
    sb ++= "# NUM_PART=%s\n".format(NUM_PART)

    sb ++= "# enableOracleCache=%s\n".format(enableOracleCache)
    sb ++= "# oracleCacheSize=%d\n".format(oracleCacheSize)

    sb ++= "# H=%d\n".format(H)
    sb ++= "# sampleFrac=%f\n".format(sampleFrac)
    sb ++= "# sampleWithReplacement=%s\n".format(sampleWithReplacement)

    sb ++= "# debugInfoPath=%s\n".format(debugInfoPath)

    sb.toString()
  }

}