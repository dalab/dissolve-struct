package ch.ethz.dalab.dissolve.optimization

import breeze.linalg._
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.regression.LabeledObject
import java.io.File
import java.io.PrintWriter


class PerceptronSolver [X, Y](
  val data: Seq[LabeledObject[X, Y]],
  val featureFn: (X, Y) => Vector[Double], // (x, y) => FeatureVector
  val lossFn: (Y, Y) => Double, // (yTruth, yPredict) => LossValue
  val oracleFn: (StructSVMModel[X, Y], X, Y) => Y, // (model, x_i, y_i) => Label
  val predictFn: (StructSVMModel[X, Y], X) => Y,
  val solverOptions: SolverOptions[X, Y]){
  
  val roundLimit = solverOptions.roundLimit
  val lambda = solverOptions.lambda
  val debugOn: Boolean = solverOptions.debug
  
  val maxOracle = oracleFn
  val phi = featureFn
  // Number of dimensions of \phi(x, y)
  val ndims: Int = phi(data(0).pattern, data(0).label).size

  // Filenames
  val lossWriterFileName = "data/debug/perceptron-loss.csv"

  /*
   * Structured Perceptron Optimizer
   */
  def optimize(): StructSVMModel[X,Y] = {
    
    val n: Int = data.length
    val d: Int = featureFn(data(0).pattern, data(0).label).size
    val model: StructSVMModel[X, Y] = new StructSVMModel(DenseVector.zeros(featureFn(data(0).pattern, data(0).label).size),
      0.0,
      DenseVector.zeros(ndims),
      featureFn,
      lossFn,
      oracleFn,
      predictFn)
    
    
    // TBA: Initialization in case of Weighted Averaging
    var wAvg: DenseVector[Double] =
      if (solverOptions.doWeightedAveraging)
        DenseVector.zeros(d)
      else null

    var debugIter = if (solverOptions.debugMultiplier == 0) {
      solverOptions.debugMultiplier = 100
      n
    } else {
      1
    }
    
    if (debugOn) {
      println("Beginning training of %d data points in %d passes with lambda=%f".format(n, roundLimit, lambda))
    }
        
    
//    val w_a = DenseVector.zeros[Double](d)
//    val c = 1
    
    for (passNum <- 0 until roundLimit){
      if (debugOn)
        println("Starting pass #%d".format(passNum))

      for (dummy <- 0 until n) {
        
        val i = dummy
        val yi = data(i).label
        val xi = data(i).pattern
        
        val y_opt = oracleFn(model, xi, yi)
        
        if (y_opt == data(i).label){
          
          val newWeights = model.getWeights() + phi(xi, yi) - phi(xi, y_opt)
          model.updateWeights(newWeights)
          println("Pass #%d Iteration #%d".format(passNum, i))
          
        }
        
      }
        
      if (debugOn)
        println("Completed pass #%d".format(passNum))
      
    }
      
    model
  }

}
