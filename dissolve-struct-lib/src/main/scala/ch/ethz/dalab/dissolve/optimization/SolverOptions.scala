package ch.ethz.dalab.dissolve.optimization

import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg.Vector
import org.apache.spark.rdd.RDD
import java.io.File

sealed trait StoppingCriterion

// Option A - Limit number of communication rounds
case object RoundLimitCriterion extends StoppingCriterion {
  override def toString(): String = { "RoundLimitCriterion" }
}

// Option B - Check gap
case object GapThresholdCriterion extends StoppingCriterion {
  override def toString(): String = { "GapThresholdCriterion" }
}

// Option C - Run for this amount of time (in secs)
case object TimeLimitCriterion extends StoppingCriterion {
  override def toString(): String = { "TimeLimitCriterion" }
}

class SolverOptions[X, Y] extends Serializable {
  var doWeightedAveraging: Boolean = false

  var randSeed: Int = 42
  /**
   *  BCFW - "uniform", "perm" or "iter"
   *  DBCFW - "count", "frac"
   */
  var sample: String = "frac"
  var lambda: Double = 0.01 // FIXME This is 1/n in Matlab code

  var testData: Option[Seq[LabeledObject[X, Y]]] = Option.empty[Seq[LabeledObject[X, Y]]]
  var testDataRDD: Option[RDD[LabeledObject[X, Y]]] = Option.empty[RDD[LabeledObject[X, Y]]]

  var doLineSearch: Boolean = true

  // Checkpoint once in these many rounds
  var checkpointFreq: Int = 50

  // In case of multi-class
  var numClasses = -1

  
  // Cache params
  var enableOracleCache: Boolean = false
  var oracleCacheSize: Int = 10

  // DBCFW specific params
  var H: Int = 5 // Number of data points to sample in each round of CoCoA (= number of local coordinate updates)
  var sampleFrac: Double = 0.5
  var sampleWithReplacement: Boolean = false

  var enableManualPartitionSize: Boolean = false
  var NUM_PART: Int = 1 // Number of partitions of the RDD

  // SSG specific params
  var ssg_gamma0: Int = 1000
  
  // For debugging/Testing purposes
  // Basic debugging flag
  var debug: Boolean = false
  // Obtain statistics (primal value, duality gap, train error, test error, etc.) once in these many rounds.
  // If 1, obtains statistics in each round
  var debugMultiplier: Int = 1

  // Option A - Limit number of communication rounds
  var roundLimit: Int = 25

  // Option B - Check gap
  var gapThreshold: Double = 0.1
  var gapCheck: Int = 1 // Check for once these many rounds

  // Option C - Run for this amount of time (in secs)
  var timeLimit: Int = 300

  var stoppingCriterion: StoppingCriterion = RoundLimitCriterion

  // Sparse representation of w_i's
  var sparse: Boolean = false

  // Path to write the CSVs
  var debugInfoPath: String = new File(".").getCanonicalPath() + "/debugInfo-%d.csv".format(System.currentTimeMillis())
  
  // Level History
  // Store last decoded level, and resume searching for candidates on that level
  var resumeMaxLevel: Boolean = false
  // Store past 10 decodings. For every decoding, check that this hasn't been 
  // already suggested. If it has, move to next level (if exists)
  var stubRepetitions: Boolean = false

  override def toString(): String = {
    val sb: StringBuilder = new StringBuilder()

    sb ++= "# numRounds=%s\n".format(roundLimit)
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

    sb ++= "# checkpointFreq=%d\n".format(checkpointFreq)

    sb ++= "# stoppingCriterion=%s\n".format(stoppingCriterion)
    this.stoppingCriterion match {
      case RoundLimitCriterion   => sb ++= "# roundLimit=%d\n".format(roundLimit)
      case GapThresholdCriterion => sb ++= "# gapThreshold=%f\n".format(gapThreshold)
      case TimeLimitCriterion    => sb ++= "# timeLimit=%d\n".format(timeLimit)
      case _                     => throw new Exception("Unrecognized Stopping Criterion")
    }

    sb ++= "# debugMultiplier=%d\n".format(debugMultiplier)
    
    sb ++= "# resumeMaxLevel=%s\n".format(resumeMaxLevel)
    sb ++= "# stubRepetitions=%s\n".format(stubRepetitions)

    sb.toString()
  }

}
