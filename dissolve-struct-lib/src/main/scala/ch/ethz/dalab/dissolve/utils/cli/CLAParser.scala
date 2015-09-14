package ch.ethz.dalab.dissolve.utils.cli

import ch.ethz.dalab.dissolve.optimization._
import scopt.OptionParser

object CLAParser {

  def getParser(): OptionParser[Config] =
    new scopt.OptionParser[Config]("spark-submit ... <jar>") {
      head("dissolve^struct", "0.1-SNAPSHOT")

      help("help") text ("prints this usage text")

      opt[Double]("lambda") action { (x, c) =>
        c.copy(lambda = x)
      } text ("Regularization constant. Default = 0.01")

      opt[Int]("randseed") action { (x, c) =>
        c.copy(randSeed = x)
      } text ("Random Seed. Default = 42")

      opt[Unit]("linesearch") action { (_, c) =>
        c.copy(lineSearch = true)
      } text ("Enable Line Search. Default = false")

      opt[Double]("samplefrac") action { (x, c) =>
        c.copy(sampleFrac = x)
      } text ("Fraction of original dataset to be sampled in each round. Default = 0.5")

      opt[Int]("minpart") action { (x, c) =>
        c.copy(minPartitions = x)
      } text ("Repartition data RDD to given number of partitions before training. Default = Auto")

      opt[Unit]("sparse") action { (_, c) =>
        c.copy(sparse = true)
      } text ("Maintain vectors as sparse vectors. Default = Dense.")

      opt[Int]("oraclecachesize") action { (x, c) =>
        c.copy(oracleCacheSize = x)
      } text ("Oracle Cache Size (caching answers of the maximization oracle for this datapoint). Default = Disabled")

      opt[Int]("cpfreq") action { (x, c) =>
        c.copy(checkpointFreq = x)
      } text ("Checkpoint Frequency (in rounds). Default = 50")

      opt[String]("stopcrit") action { (x, c) =>
        x match {
          case "round" =>
            c.copy(stoppingCriterion = RoundLimitCriterion)
          case "gap" =>
            c.copy(stoppingCriterion = GapThresholdCriterion)
          case "time" =>
            c.copy(stoppingCriterion = TimeLimitCriterion)
        }
      } validate { x =>
        x match {
          case "round" => success
          case "gap"   => success
          case "time"  => success
          case _       => failure("Stopping criterion has to be one of: round | gap | time")
        }
      } text ("Stopping Criterion. (round | gap | time). Default = round")

      opt[Int]("roundlimit") action { (x, c) =>
        c.copy(roundLimit = x)
      } text ("Round Limit. Default = 25")

      opt[Double]("gapthresh") action { (x, c) =>
        c.copy(gapThreshold = x)
      } text ("Gap Threshold. Default = 0.1")

      opt[Int]("gapcheck") action { (x, c) =>
        c.copy(gapCheck = x)
      } text ("Checks for gap every these many rounds. Default = 25")

      opt[Int]("timelimit") action { (x, c) =>
        c.copy(timeLimit = x)
      } text ("Time Limit (in secs). Default = 300 secs")

      opt[Unit]("debug") action { (_, c) =>
        c.copy(debug = true)
      } text ("Enable debugging. Default = false")

      opt[Int]("debugmult") action { (x, c) =>
        c.copy(debugMultiplier = x)
      } text ("Frequency of debugging. Obtains gap, train and test errors. Default = 1")

      opt[String]("debugfile") action { (x, c) =>
        c.copy(debugPath = x)
      } text ("Path to debug file. Default = current-dir")
      
      opt[Unit]("resumelevel") action { (_, c) =>
        c.copy(resumeMaxLevel = true)
      } text ("Resumes decoding at previous level. Default = false")
      
      opt[Unit]("stubrepeat") action { (_, c) =>
        c.copy(stubRepetitions = true)
      } text ("Keeps track of previous 10 decodings. Decodes at next level in case a decoding is repeated. Default = false")

      opt[Map[String, String]]("kwargs") valueName ("k1=v1,k2=v2...") action { (x, c) =>
        c.copy(kwargs = x)
      } text ("other arguments")
    }

  def argsToOptions[X, Y](args: Array[String]): (SolverOptions[X, Y], Map[String, String]) =

    getParser().parse(args, Config()) match {
      case Some(config) =>
        val solverOptions: SolverOptions[X, Y] = new SolverOptions[X, Y]()
        // Copy all config parameters to a Solver Options instance
        solverOptions.lambda = config.lambda
        solverOptions.randSeed = config.randSeed
        solverOptions.doLineSearch = config.lineSearch
        solverOptions.doWeightedAveraging = config.wavg

        solverOptions.sampleFrac = config.sampleFrac
        if (config.minPartitions > 0) {
          solverOptions.enableManualPartitionSize = true
          solverOptions.NUM_PART = config.minPartitions
        }
        solverOptions.sparse = config.sparse

        if (config.oracleCacheSize > 0) {
          solverOptions.enableOracleCache = true
          solverOptions.oracleCacheSize = config.oracleCacheSize
        }

        solverOptions.checkpointFreq = config.checkpointFreq

        solverOptions.stoppingCriterion = config.stoppingCriterion
        solverOptions.roundLimit = config.roundLimit
        solverOptions.gapCheck = config.gapCheck
        solverOptions.gapThreshold = config.gapThreshold
        solverOptions.timeLimit = config.timeLimit

        solverOptions.debug = config.debug
        solverOptions.debugMultiplier = config.debugMultiplier
        solverOptions.debugInfoPath = config.debugPath
        
        solverOptions.resumeMaxLevel = config.resumeMaxLevel
        solverOptions.stubRepetitions = config.stubRepetitions

        (solverOptions, config.kwargs)

      case None =>
        // No options passed. Do nothing.
        val solverOptions: SolverOptions[X, Y] = new SolverOptions[X, Y]()
        val kwargs = Map[String, String]()

        (solverOptions, kwargs)
    }

  def main(args: Array[String]): Unit = {
    val foo = argsToOptions(args)
    println(foo._1.toString())
    println(foo._2)
  }

}