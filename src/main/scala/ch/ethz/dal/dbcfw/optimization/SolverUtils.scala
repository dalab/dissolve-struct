package ch.ethz.dal.dbcfw.optimization

import ch.ethz.dal.dbcfw.classification.StructSVMModel
import breeze.linalg._
import ch.ethz.dal.dbcfw.regression.LabeledObject

object SolverUtils {

  /**
   * Average loss
   */
  def averageLoss(data: Vector[LabeledObject],
    lossFn: (Vector[Double], Vector[Double]) => Double,
    predictFn: (StructSVMModel, Matrix[Double]) => Vector[Double],
    model: StructSVMModel): Double = {

    var lossTerm: Double = 0.0
    for (i <- 0 until data.size) {
      val ystar_i = predictFn(model, data(i).pattern)
      lossTerm += lossFn(data(i).label, ystar_i)
    }

    // Return average of loss terms
    lossTerm / (data.size.toDouble)
  }

  /**
   * Objective function
   */
  def objectiveFunction(w: Vector[Double],
    b_alpha: Double,
    lambda: Double): Double = {
    // Return the value of f(alpha)
    0.5 * lambda * (w.t * w) - b_alpha
  }

  /**
   * Compute Duality gap
   */
  def dualityGap(data: Vector[LabeledObject],
    featureFn: (Vector[Double], Matrix[Double]) => Vector[Double],
    lossFn: (Vector[Double], Vector[Double]) => Double,
    oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double],
    model: StructSVMModel,
    lambda: Double): (Double, DenseVector[Double], Double) = {

    val phi = featureFn
    val maxOracle = oracleFn

    val w: Vector[Double] = model.getWeights()
    val ell: Double = model.getEll()

    val n: Int = data.size
    val d: Int = model.getWeights().size
    val yStars = new Array[Vector[Double]](n)

    for (i <- 0 until n) {
      yStars(i) = maxOracle(model, data(i).label, data(i).pattern)
    }

    var w_s: DenseVector[Double] = DenseVector.zeros[Double](d)
    var ell_s: Double = 0.0
    for (i <- 0 until n) {
      w_s += phi(data(i).label, data(i).pattern) - phi(yStars(i), data(i).pattern)
      ell_s += lossFn(data(i).label, yStars(i))
    }

    w_s = w_s / (lambda * n)
    ell_s = ell_s / n

    val gap: Double = w.t * (w - w_s) * lambda - ell + ell_s

    (gap, w_s, ell_s)
  }

  /**
   * Primal objective
   */
  def primalObjective(data: Vector[LabeledObject],
    featureFn: (Vector[Double], Matrix[Double]) => Vector[Double],
    lossFn: (Vector[Double], Vector[Double]) => Double,
    oracleFn: (StructSVMModel, Vector[Double], Matrix[Double]) => Vector[Double],
    model: StructSVMModel,
    lambda: Double): Double = {

    var hingeLosses: Double = 0.0
    for (i <- 0 until data.size) {
      val yStar_i = oracleFn(model, data(i).label, data(i).pattern)
      val loss_i = lossFn(data(i).label, yStar_i)
      val psi_i = featureFn(data(i).label, data(i).pattern) - featureFn(yStar_i, data(i).pattern)

      val hingeloss_i = loss_i - model.getWeights().t * psi_i
      // println("loss_i = %f, other_loss = %f".format(loss_i, model.getWeights().t * psi_i))
      // assert(hingeloss_i >= 0.0)

      hingeLosses += hingeloss_i
    }

    // Compute the primal and return it
    0.5 * lambda * (model.getWeights.t * model.getWeights) + hingeLosses / data.size

  }

}