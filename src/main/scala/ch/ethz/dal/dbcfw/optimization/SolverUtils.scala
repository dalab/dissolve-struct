package ch.ethz.dal.dbcfw.optimization

import ch.ethz.dal.dbcfw.classification.StructSVMModel
import breeze.linalg._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import scala.reflect.ClassTag

object SolverUtils {

  /**
   * Average loss
   */
  def averageLoss[X, Y](data: Vector[LabeledObject[X, Y]],
    lossFn: (Y, Y) => Double,
    predictFn: (StructSVMModel[X, Y], X) => Y,
    model: StructSVMModel[X, Y]): Double = {

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
  def dualityGap[X, Y](data: Vector[LabeledObject[X, Y]],
    featureFn: (Y, X) => Vector[Double],
    lossFn: (Y, Y) => Double,
    oracleFn: (StructSVMModel[X, Y], Y, X) => Y,
    model: StructSVMModel[X, Y],
    lambda: Double)(implicit m: ClassTag[Y]): (Double, Vector[Double], Double) = {

    val phi = featureFn
    val maxOracle = oracleFn

    val w: Vector[Double] = model.getWeights()
    val ell: Double = model.getEll()

    val n: Int = data.size
    val d: Int = model.getWeights().size
    val yStars = new Array[Y](n)

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
  def primalObjective[X, Y](data: Vector[LabeledObject[X, Y]],
    featureFn: (Y, X) => Vector[Double],
    lossFn: (Y, Y) => Double,
    oracleFn: (StructSVMModel[X, Y], Y, X) => Y,
    model: StructSVMModel[X, Y],
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

  /**
   * Get Spark's properties
   */
  def getSparkConfString(sc: SparkConf): String = {
    val keys = List("spark.app.name", "spark.executor.memory", "spark.task.cpus", "spark.local.dir", "spark.default.parallelism")
    val sb: StringBuilder = new StringBuilder()

    for (key <- keys)
      sb ++= "# %s=%s\n".format(key, sc.get(key, "NA"))

    sb.toString()
  }

}