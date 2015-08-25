package ch.ethz.dalab.dissolve.optimization

import ch.ethz.dalab.dissolve.classification.StructSVMModel
import breeze.linalg._
import ch.ethz.dalab.dissolve.regression.LabeledObject
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import scala.reflect.ClassTag

object SolverUtils {

  /**
   * Average loss
   */
  def averageLoss[X, Y](data: Seq[LabeledObject[X, Y]],
                        dissolveFunctions: DissolveFunctions[X, Y],
                        model: StructSVMModel[X, Y]): (Double, Double) = {

    var errorTerm: Double = 0.0
    var structuredHingeLoss: Double = 0.0

    for (i <- 0 until data.size) {
      val ystar_i = dissolveFunctions.predictFn(model, data(i).pattern)
      val loss = dissolveFunctions.lossFn(ystar_i, data(i).label)
      errorTerm += loss

      val wFeatureDotProduct = model.getWeights().t * dissolveFunctions.featureFn(data(i).pattern, data(i).label)
      val structuredHingeLoss: Double = loss - wFeatureDotProduct
    }

    // Return average of loss terms
    (errorTerm / (data.size.toDouble), structuredHingeLoss / (data.size.toDouble))
  }

  def averageLoss[X, Y](data: RDD[LabeledObject[X, Y]],
                        dissolveFunctions: DissolveFunctions[X, Y],
                        model: StructSVMModel[X, Y],
                        dataSize: Int): (Double, Double) = {

    val (loss, hloss) =
      data.map {
        case datapoint =>
          val ystar_i = dissolveFunctions.predictFn(model, datapoint.pattern)
          val loss = dissolveFunctions.lossFn(ystar_i, datapoint.label)
          val wFeatureDotProduct = model.getWeights().t * (dissolveFunctions.featureFn(datapoint.pattern, datapoint.label)
            - dissolveFunctions.featureFn(datapoint.pattern, ystar_i))
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
                       oracleFn: (StructSVMModel[X, Y], X, Y) => Y,
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
      yStars(i) = maxOracle(model, data(i).pattern, data(i).label)
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
                       model: StructSVMModel[X, Y],
                       lambda: Double,
                       dataSize: Int)(implicit m: ClassTag[Y]): (Double, Vector[Double], Double) = {

    val phi = dissolveFunctions.featureFn _
    val maxOracle = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _

    val w: Vector[Double] = model.getWeights()
    val ell: Double = model.getEll()

    val n: Int = dataSize.toInt
    val d: Int = model.getWeights().size

    var (w_s, ell_s) = data.map {
      case datapoint =>
        val yStar = maxOracle(model, datapoint.pattern, datapoint.label)
        val w_s = phi(datapoint.pattern, datapoint.label) - phi(datapoint.pattern, yStar)
        val ell_s = lossFn(yStar, datapoint.label)

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
  def primalObjective[X, Y](data: Vector[LabeledObject[X, Y]],
                            dissolveFunctions: DissolveFunctions[X, Y],
                            model: StructSVMModel[X, Y],
                            lambda: Double): Double = {

    val featureFn = dissolveFunctions.featureFn _
    val oracleFn = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _

    var hingeLosses: Double = 0.0
    for (i <- 0 until data.size) {
      val yStar_i = oracleFn(model, data(i).pattern, data(i).label)
      val loss_i = lossFn(yStar_i, data(i).label)
      val psi_i = featureFn(data(i).pattern, data(i).label) - featureFn(data(i).pattern, yStar_i)

      val hingeloss_i = loss_i - model.getWeights().t * psi_i
      // println("loss_i = %f, other_loss = %f".format(loss_i, model.getWeights().t * psi_i))
      // assert(hingeloss_i >= 0.0)

      hingeLosses += hingeloss_i
    }

    // Compute the primal and return it
    0.5 * lambda * (model.getWeights.t * model.getWeights) + hingeLosses / data.size

  }

  type TotalPixelCount = Int
  type CorrectLabelingCount = Int
  case class DataEval(gap: Double,
                      avgDelta: Double,
                      avgHLoss: Double,
                      perClassAccuracy: Array[Double],
                      globalAccuracy: Double)

  case class PartialTrainDataEval(sum_w_s: Vector[Double],
                                  sum_ell_s: Double,
                                  sum_Delta: Double,
                                  sum_HLoss: Double,
                                  sum_PerClassAccuracy: Array[(TotalPixelCount, CorrectLabelingCount)])
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
                          model: StructSVMModel[X, Y],
                          lambda: Double,
                          dataSize: Int)(implicit m: ClassTag[Y]): DataEval = {

    val phi = dissolveFunctions.featureFn _
    val maxOracle = dissolveFunctions.oracleFn _
    val lossFn = dissolveFunctions.lossFn _
    val predictFn = dissolveFunctions.predictFn _
    val perClassAccuracy = dissolveFunctions.perClassAccuracy _

    val numClasses = dissolveFunctions.numClasses()

    val w: Vector[Double] = model.getWeights()
    val ell: Double = model.getEll()

    val n: Int = dataSize.toInt
    val d: Int = model.getWeights().size

    def twoTupArraySum(arrA: Array[(Int, Int)],
                       arrB: Array[(Int, Int)]): Array[(Int, Int)] =
      {
        assert(arrA.size == arrB.size)
        assert(arrA.size == numClasses)

        arrA
          .zip(arrB)
          .map {
            case ((totCountA, correctCountA), (totCountB, correctCountB)) =>
              (totCountA + totCountB, correctCountA + correctCountB)
          }
      }

    val initEval =
      PartialTrainDataEval(Vector.zeros[Double](d),
        0.0,
        0.0,
        0.0,
        Array.fill(numClasses)((0, 0)))

    val partialEval = data.map {
      case datapoint =>
        /**
         * Gap and Structured HingeLoss
         */
        val lossAug_yStar = maxOracle(model, datapoint.pattern, datapoint.label)
        val w_s = phi(datapoint.pattern, datapoint.label) - phi(datapoint.pattern, lossAug_yStar)
        val ell_s = lossFn(lossAug_yStar, datapoint.label)
        val lossAug_wFeatureDotProduct = lossFn(lossAug_yStar, datapoint.label) -
          (model.getWeights().t * (phi(datapoint.pattern, datapoint.label)
            - phi(datapoint.pattern, lossAug_yStar)))
        val structuredHingeLoss: Double = lossAug_wFeatureDotProduct

        /**
         * \Delta
         */
        val predict_yStar = predictFn(model, datapoint.pattern)
        val loss = lossFn(predict_yStar, datapoint.label)

        /**
         * Per-class loss
         */
        val y_truth = datapoint.label
        val y_predicted = predict_yStar
        val y_perClassLoss: Array[(TotalPixelCount, CorrectLabelingCount)] =
          perClassAccuracy(y_predicted, y_truth)

        PartialTrainDataEval(w_s,
          ell_s,
          loss,
          structuredHingeLoss,
          y_perClassLoss)

    }.fold(initEval) {
      case (prevEval, nextEval) =>

        val sum_w_s = prevEval.sum_w_s + nextEval.sum_w_s
        val sum_ell_s: Double = prevEval.sum_ell_s + nextEval.sum_ell_s
        val sum_Delta: Double = prevEval.sum_Delta + nextEval.sum_Delta
        val sum_HLoss: Double = prevEval.sum_HLoss + nextEval.sum_HLoss
        val sum_PerClassError: Array[(TotalPixelCount, CorrectLabelingCount)] =
          twoTupArraySum(prevEval.sum_PerClassAccuracy,
            nextEval.sum_PerClassAccuracy)

        PartialTrainDataEval(sum_w_s,
          sum_ell_s,
          sum_Delta,
          sum_HLoss,
          sum_PerClassError)

    }

    // Gap
    val sum_w_s = partialEval.sum_w_s
    val w_s = sum_w_s / (lambda * n)
    val ell_s = partialEval.sum_ell_s / n
    val gap: Double = w.t * (w - w_s) * lambda - ell + ell_s

    // Loss
    val avgLoss = partialEval.sum_Delta / n
    val avgHLoss = partialEval.sum_HLoss / n

    // Per-pixel errors
    // A. Per Class Errors
    val backgroundLabel = numClasses - 1 // Assumes black is last label
    val avgPerClassError = partialEval.sum_PerClassAccuracy.map {
      case (totalPixelCount, correctLabelCount) =>
        if (totalPixelCount > 0)
          correctLabelCount.toDouble / totalPixelCount.toDouble
        else
          // Represent unencountered classes as 0.0
          // Then, when calculating the accuracy, drop these with 0-accuracy
          0.0
    }
    // B. Global error
    val numPixels: Long = partialEval
      .sum_PerClassAccuracy
      .dropRight(1) // Assumes Background label is the last one
      .map(_._1.toLong)
      .sum
    val numCorrectLabels = partialEval
      .sum_PerClassAccuracy
      .dropRight(1) // Assumes Background label is the last one
      .map(_._2.toLong)
      .sum

    val globalError = numCorrectLabels.toDouble / numPixels

    DataEval(gap,
      avgLoss,
      avgHLoss,
      avgPerClassError,
      globalError)
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