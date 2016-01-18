package ch.ethz.dalab.dissolve.optimization

import ch.ethz.dalab.dissolve.classification.MutableWeightsEll
import breeze.linalg._
import ch.ethz.dalab.dissolve.regression.LabeledObject
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import scala.reflect.ClassTag
import org.apache.spark.broadcast.Broadcast

object SolverUtils {

  /**
   * Average loss
   */
  def averageLoss[X, Y](data: Seq[LabeledObject[X, Y]],
                        dissolveFunctions: DissolveFunctions[X, Y],
                        model: MutableWeightsEll): (Double, Double) = {

    var errorTerm: Double = 0.0
    var structuredHingeLoss: Double = 0.0

    for (i <- 0 until data.size) {
      val ystar_i = dissolveFunctions.predictFn(model.getWeights(), data(i).pattern)
      val loss = dissolveFunctions.lossFn(data(i).label, ystar_i)
      errorTerm += loss

      val wFeatureDotProduct = model.getWeights().t * dissolveFunctions.featureFn(data(i).pattern, data(i).label)
      val structuredHingeLoss: Double = loss - wFeatureDotProduct
    }

    // Return average of loss terms
    (errorTerm / (data.size.toDouble), structuredHingeLoss / (data.size.toDouble))
  }

  def averageLoss[X, Y](data: RDD[LabeledObject[X, Y]],
                        dissolveFunctions: DissolveFunctions[X, Y],
                        model: MutableWeightsEll,
                        dataSize: Int): (Double, Double) = {

    val featureFn = dissolveFunctions.featureFn _
    val predictFn = dissolveFunctions.predictFn _
    val oracleFn = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _
    val classWeight = dissolveFunctions.classWeights _

    val (loss, hloss) =
      data.map {
        case datapoint =>
          val xi = datapoint.pattern
          val yi = datapoint.label
          
          val ystar_i = predictFn(model.getWeights(), xi)
          val loss = lossFn(ystar_i, datapoint.label)

          val wFeatureDotProduct = model.getWeights().t * (featureFn(xi, yi)
            - featureFn(xi, ystar_i))
          val structuredHingeLoss: Double = loss - wFeatureDotProduct

          (loss, structuredHingeLoss)
      }.fold((0.0, 0.0)) {
        case ((lossAccum, hlossAccum), (loss, hloss)) =>
          (lossAccum + loss, hlossAccum + hloss)
      }

    (loss / dataSize, hloss / dataSize)
  }

  /**
   * Objective function (SVM dual, assuming we know the vector b of all losses. See BCFW paper)
   */
  def objectiveFunction(w: Vector[Double],
                        b_alpha: Double,
                        lambda: Double): Double = {
    // Return the value of f(alpha)
    0.5 * lambda * (w.t * w) - b_alpha
  }

  /**
   * Compute Duality gap
   * Requires one full pass of decoding over all data examples.
   */
  def dualityGap[X, Y](data: Seq[LabeledObject[X, Y]],
                       featureFn: (X, Y) => Vector[Double],
                       lossFn: (Y, Y) => Double,
                       oracleFn: (Vector[Double], X, Y) => Y,
                       model: MutableWeightsEll,
                       lambda: Double)(implicit m: ClassTag[Y]): (Double, Vector[Double], Double) = {

    val phi = featureFn
    val maxOracle = oracleFn

    val w: Vector[Double] = model.getWeights()
    val ell: Double = model.getEll()

    val n: Int = data.size
    val d: Int = model.getWeights().size
    val yStars = new Array[Y](n)

    for (i <- 0 until n) {
      yStars(i) = maxOracle(model.getWeights(), data(i).pattern, data(i).label)
    }

    var w_s: DenseVector[Double] = DenseVector.zeros[Double](d)
    var ell_s: Double = 0.0
    for (i <- 0 until n) {
      w_s += phi(data(i).pattern, data(i).label) - phi(data(i).pattern, yStars(i))
      ell_s += lossFn(yStars(i), data(i).label)
    }

    w_s = w_s / (lambda * n)
    ell_s = ell_s / n

    val gap: Double = w.t * (w - w_s) * lambda - ell + ell_s

    (gap, w_s, ell_s)
  }

  /**
   * Alternative implementation, using fold. TODO: delete this or the above
   * Requires one full pass of decoding over all data examples.
   */
  def dualityGap[X, Y](data: RDD[LabeledObject[X, Y]],
                       dissolveFunctions: DissolveFunctions[X, Y],
                       model: MutableWeightsEll,
                       lambda: Double,
                       dataSize: Int)(implicit m: ClassTag[Y]): (Double, Vector[Double], Double) = {

    val phi = dissolveFunctions.featureFn _
    val maxOracle = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _
    val classWeight = dissolveFunctions.classWeights _

    val w: Vector[Double] = model.getWeights()
    val ell: Double = model.getEll()

    val n: Int = dataSize.toInt
    val d: Int = model.getWeights().size

    var (w_s, ell_s) = data.map {
      case datapoint =>
        val yStar = maxOracle(model.getWeights(), datapoint.pattern, datapoint.label)
        val w_s = (phi(datapoint.pattern, datapoint.label) - phi(datapoint.pattern, yStar)) * classWeight(datapoint.label)
        val ell_s = lossFn(yStar, datapoint.label) * classWeight(datapoint.label)

        (w_s, ell_s)
    }.fold((Vector.zeros[Double](d), 0.0)) {
      case ((w_acc, ell_acc), (w_i, ell_i)) =>
        (w_acc + w_i, ell_acc + ell_i)
    }

    w_s = w_s / (lambda * n)
    ell_s = ell_s / n

    val gap: Double = w.t * (w - w_s) * lambda - ell + ell_s

    (gap, w_s, ell_s)
  }

  /**
   * Primal objective.
   * Requires one full pass of decoding over all data examples.
   */
  def primalObjective[X, Y](data: Seq[LabeledObject[X, Y]],
                            dissolveFunctions: DissolveFunctions[X, Y],
                            model: MutableWeightsEll,
                            lambda: Double): Double = {

    val featureFn = dissolveFunctions.featureFn _
    val oracleFn = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _
    val classWeight = dissolveFunctions.classWeights _

    var hingeLosses: Double = 0.0
    for (i <- 0 until data.size) {
      val yStar_i = oracleFn(model.getWeights(), data(i).pattern, data(i).label)
      val loss_i = lossFn(yStar_i, data(i).label) * classWeight(data(i).label)
      val psi_i = featureFn(data(i).pattern, data(i).label) - featureFn(data(i).pattern, yStar_i) * classWeight(data(i).label)

      val hingeloss_i = loss_i - model.getWeights().t * psi_i
      // println("loss_i = %f, other_loss = %f".format(loss_i, model.getWeights().t * psi_i))
      // assert(hingeloss_i >= 0.0)

      hingeLosses += hingeloss_i
    }

    // Compute the primal and return it
    0.5 * lambda * (model.getWeights.t * model.getWeights) + hingeLosses / data.size

  }

  /**
   * Primal objective.
   * Requires one full pass of decoding over all data examples.
   */
  def primalObjective[X, Y](data: RDD[LabeledObject[X, Y]],
                            dataSize: Int,
                            dissolveFunctions: DissolveFunctions[X, Y],
                            model: MutableWeightsEll,
                            lambda: Double): (Double, Double) = {

    val featureFn = dissolveFunctions.featureFn _
    val oracleFn = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _
    val classWeight = dissolveFunctions.classWeights _

    val hingeLossSum = data.map {
      case lo =>
        val xi = lo.pattern
        val yi = lo.label

        val yStar_i = oracleFn(model.getWeights(), xi, yi)
        val loss_i = lossFn(yStar_i, yi) * classWeight(yi)
        val psi_i = featureFn(xi, yi) - featureFn(xi, yStar_i) * classWeight(yi)

        val hingeloss_i = loss_i - model.getWeights().t * psi_i

        hingeloss_i
    }.reduce(_ + _)

    val meanHingeLoss = hingeLossSum / dataSize
    // Compute the primal and return it
    val primal = 0.5 * lambda * (model.getWeights.t * model.getWeights) + meanHingeLoss

    (primal, meanHingeLoss)
  }

  case class DataEval(gap: Double,
                      avgDelta: Double,
                      avgHLoss: Double)

  case class PartialTrainDataEval(sum_w_s: Vector[Double],
                                  sum_ell_s: Double,
                                  sum_Delta: Double,
                                  sum_HLoss: Double) {

    def +(that: PartialTrainDataEval): PartialTrainDataEval = {

      val sum_w_s = this.sum_w_s + that.sum_w_s
      val sum_ell_s: Double = this.sum_ell_s + that.sum_ell_s
      val sum_Delta: Double = this.sum_Delta + that.sum_Delta
      val sum_HLoss: Double = this.sum_HLoss + that.sum_HLoss

      PartialTrainDataEval(sum_w_s,
        sum_ell_s,
        sum_Delta,
        sum_HLoss)

    }
  }
  /**
   * Makes an additional pass over the data to compute the following:
   * 1. Duality Gap
   * 2. Average \Delta
   * 3. Average Structured Hinge Loss
   * 4. Average Per-Class pixel-loss
   * 5. Global loss
   */
  def trainDataEval[X, Y](data: RDD[LabeledObject[X, Y]],
                          dissolveFunctions: DissolveFunctions[X, Y],
                          model: MutableWeightsEll,
                          lambda: Double,
                          dataSize: Int)(implicit m: ClassTag[Y]): DataEval = {

    val phi = dissolveFunctions.featureFn _
    val maxOracle = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _
    val predictFn = dissolveFunctions.predictFn _
    val classWeight = dissolveFunctions.classWeights _

    val n: Int = dataSize.toInt
    val d: Int = model.getWeights().size

    val bcModel: Broadcast[MutableWeightsEll] = data.context.broadcast(model)

    val initEval =
      PartialTrainDataEval(DenseVector.zeros[Double](d),
        0.0,
        0.0,
        0.0)

    val partialEval = data.map {
      case datapoint =>
        /**
         * Gap and Structured HingeLoss
         */
        val lossAug_yStar = maxOracle(bcModel.value.getWeights(), datapoint.pattern, datapoint.label)
        val w_s = (phi(datapoint.pattern, datapoint.label) - phi(datapoint.pattern, lossAug_yStar)) * classWeight(datapoint.label)
        val ell_s = lossFn(datapoint.label, lossAug_yStar) * classWeight(datapoint.label)
        val lossAug_wFeatureDotProduct = lossFn(datapoint.label, lossAug_yStar) -
          (bcModel.value.getWeights().t * (phi(datapoint.pattern, datapoint.label)
            - phi(datapoint.pattern, lossAug_yStar)))
        val structuredHingeLoss: Double = lossAug_wFeatureDotProduct

        /**
         * \Delta
         */
        val predict_yStar = predictFn(bcModel.value.getWeights(), datapoint.pattern)
        val loss = lossFn(datapoint.label, predict_yStar)

        /**
         * Per-class loss
         */
        val y_truth = datapoint.label
        val y_predicted = predict_yStar

        PartialTrainDataEval(w_s,
          ell_s,
          loss,
          structuredHingeLoss)

    }.reduce(_ + _)

    val w: Vector[Double] = model.getWeights()
    val ell: Double = model.getEll()

    // Gap
    val sum_w_s = partialEval.sum_w_s
    val w_s = sum_w_s / (lambda * n)
    val ell_s = partialEval.sum_ell_s / n
    val gap: Double = w.t * (w - w_s) * lambda - ell + ell_s

    // Loss
    val avgLoss = partialEval.sum_Delta / n
    val avgHLoss = partialEval.sum_HLoss / n

    DataEval(gap,
      avgLoss,
      avgHLoss)
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