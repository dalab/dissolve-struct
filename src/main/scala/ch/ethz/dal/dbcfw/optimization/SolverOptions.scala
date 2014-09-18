package ch.ethz.dal.dbcfw.optimization

import ch.ethz.dal.dbcfw.regression.LabeledObject

import breeze.linalg.Vector

class SolverOptions extends Serializable {
  var numPasses: Int = 50
  var doWeightedAveraging: Boolean = true
  var timeBudget = Int.MaxValue
  
  var randSeed: Int = 1
  var sample: String = "perm" // "uniform", "perm" or "iter"
  var debugMultiplier: Int = 0
  var lambda: Double = 0.01 // FIXME This is 1/n in Matlab code
  var testData: Vector[LabeledObject] = null
  var doLineSearch: Boolean = false
  
  // DBCFW specific params
  var autoconfigure: Boolean = false
  var H: Int = 0 // Number of points to sample in each pass
  var NUM_PART: Int = 1 // Number of paritions of the RDD
  var NUM_ROUNDS: Int = 5 // Number of communication rounds
  
  // For debugging/Testing purposes
  // Basic debugging flag
  var debug: Boolean = false
  // More verbose debugging flag
  var xldebug: Boolean = false
  // Write weights to CSV after each pass
  var debugWeights: Boolean = false
  // Dump loss through iterations
  var debugLoss: Boolean = true
}