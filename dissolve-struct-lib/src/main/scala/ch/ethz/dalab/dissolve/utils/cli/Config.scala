package ch.ethz.dalab.dissolve.utils.cli

import ch.ethz.dalab.dissolve.optimization._
import java.io.File

case class Config(

  // BCFW parameters
  lambda: Double = 0.01,
  randSeed: Int = 42,
  lineSearch: Boolean = false,
  wavg: Boolean = false,

  // dissolve^struct parameters
  sampleFrac: Double = 0.5,
  minPartitions: Int = 0,
  sparse: Boolean = false,

  // Oracle
  oracleCacheSize: Int = 0,

  // Spark
  checkpointFreq: Int = 50,

  // Stopping criteria
  stoppingCriterion: StoppingCriterion = RoundLimitCriterion,
  // A - RoundLimit
  roundLimit: Int = 25,
  // B - Gap Check
  gapThreshold: Double = 0.1,
  gapCheck: Int = 10,
  // C - Time Limit
  timeLimit: Int = 300, // (In seconds)

  // Debug parameters
  debug: Boolean = false,
  debugMultiplier: Int = 1,
  debugPath: String = new File(".", "debug-%d.csv".format(System.currentTimeMillis())).getAbsolutePath,
  
  // HADES related
  resumeMaxLevel: Boolean = false,
  stubRepetitions: Boolean = false,

  // Other parameters
  kwargs: Map[String, String] = Map())