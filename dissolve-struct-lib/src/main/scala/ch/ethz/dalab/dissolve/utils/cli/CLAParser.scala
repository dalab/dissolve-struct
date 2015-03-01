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

      opt[Int]("oraclesize") action { (x, c) =>
        c.copy(oracleSize = x)
      } text ("Oracle Size. Oracle. Default = Disabled")

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

      opt[Int]("gapcheck") action { (x, c) =>
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

      opt[Map[String, String]]("kwargs") valueName ("k1=v1,k2=v2...") action { (x, c) =>
        c.copy(kwargs = x)
      } text ("other arguments")
    }

  def argsToOptions[X, Y](args: Array[String]): Unit = {
    val solverOptions: SolverOptions[X, Y] = new SolverOptions[X, Y]()

    getParser().parse(args, Config()) match {
      case Some(config) =>
        println(config)

      case None =>
        println("None")
    }
  }

  def main(args: Array[String]): Unit = {
    val foo = getParser().parse(args, Config()) match {
      case Some(config) =>
        println(config)

      case None =>
        println("None")
    }
  }

}