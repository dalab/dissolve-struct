package ch.ethz.dal.dbcfw.optimization

import ch.ethz.dal.dbcfw.regression.LabeledObject

class SolverOptions {
  var numPasses: Int = 50
  var doWeightedAveraging: Boolean = true
  var timeBudget = Int.MaxValue
  var debug: Boolean = false
  var randSeed: Int = 1
  var sample: String = "iter" // "uniform", "perm" or "iter"
  var debugMultiplier: Int = 0
  var lambda: Double = 0.01 // FIXME This is 1/n in Matlab code
  var testData: Vector[LabeledObject] = null

  var xldebug: Boolean = true
}