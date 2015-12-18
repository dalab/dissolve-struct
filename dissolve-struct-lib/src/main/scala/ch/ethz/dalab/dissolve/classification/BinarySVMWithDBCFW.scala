
/**
 *
 */
package ch.ethz.dalab.dissolve.classification

import java.io.FileWriter
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import breeze.linalg.DenseVector
import breeze.linalg.SparseVector
import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.optimization.DBCFWSolverTuned
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg.VectorBuilder
import scala.collection.mutable.HashMap
import org.apache.spark.rdd.PairRDDFunctions

/**
 * @author tribhu
 *
 */
object BinarySVMWithDBCFW extends DissolveFunctions[Vector[Double], Double] {

  val labelToWeight = HashMap[Double, Double]()

  override def classWeights(label: Double): Double = {
    labelToWeight.get(label).getOrElse(3.0)
  }

  def generateClassWeights(data: RDD[LabeledPoint]): Unit = {
    val labels: Array[Double] = data.map { x => x.label }.distinct().collect()

    val classOccur: PairRDDFunctions[Double, Double] = data.map(x => (x.label, 1.0))
    val labelOccur: PairRDDFunctions[Double, Double] = classOccur.reduceByKey((x, y) => x + y)
    val labelWeight: PairRDDFunctions[Double, Double] = labelOccur.mapValues { x => 1 / x }

    val weightSum: Double = labelWeight.values.sum()
    val nClasses: Int = 2
    val scaleValue: Double = nClasses / weightSum

    for ((label, weight) <- labelWeight.collectAsMap()) {
      labelToWeight.put(label, scaleValue * weight)
    }
  }

  /**
   * Feature function
   *
   * Analogous to phi(y) in (2)
   * Returns y_i * x_i
   *
   */
  def featureFn(x: Vector[Double], y: Double): Vector[Double] = {
    x * y
  }

  /**
   * Loss function
   *
   * Returns 0 if yTruth == yPredict, 1 otherwise
   * Equivalent to max(0, 1 - y w^T x)
   */
  def lossFn(yTruth: Double, yPredict: Double): Double =
    if (yTruth == yPredict)
      0.0
    else
      1.0

  /**
   * Maximization Oracle
   *
   * Want: max L(y_i, y) - <w, psi_i(y)>
   * This returns the most violating (Loss-augmented) label.
   */
  override def oracleFn(weights: Vector[Double], xi: Vector[Double], yi: Double): Double = {

    var score_neg1 = weights dot featureFn(xi, -1.0)
    var score_pos1 = weights dot featureFn(xi, 1.0)

    // Loss augment the scores
    score_neg1 += 1.0
    score_pos1 += 1.0

    if (yi == -1.0)
      score_neg1 -= 1.0
    else if (yi == 1.0)
      score_pos1 -= 1.0
    else
      throw new IllegalArgumentException("yi not in [-1, 1], yi = " + yi)

    if (score_neg1 > score_pos1)
      -1.0
    else
      1.0
  }

  /**
   * Prediction function
   */
  def predictFn(weights: Vector[Double], xi: Vector[Double]): Double = {

    val score_neg1 = weights dot featureFn(xi, -1.0)
    val score_pos1 = weights dot featureFn(xi, 1.0)

    if (score_neg1 > score_pos1)
      -1.0
    else
      +1.0

  }

  /**
   * Classifying with in-built functions
   */
  def train(
    data: RDD[LabeledPoint],
    solverOptions: SolverOptions[Vector[Double], Double]): StructSVMModel[Vector[Double], Double] = {

    train(data, this, solverOptions)

  }

  /**
   * Classifying with user-submitted functions
   */
  def train(
    data: RDD[LabeledPoint],
    dissolveFunctions: DissolveFunctions[Vector[Double], Double],
    solverOptions: SolverOptions[Vector[Double], Double]): StructSVMModel[Vector[Double], Double] = {

    if (solverOptions.classWeights) {
      generateClassWeights(data)
    }

    // Convert the RDD[LabeledPoint] to RDD[LabeledObject]
    val objectifiedData: RDD[LabeledObject[Vector[Double], Double]] =
      data.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], Double](x.label,
            if (solverOptions.sparse) {
              val features: Vector[Double] = x.features match {
                case features: org.apache.spark.mllib.linalg.SparseVector =>
                  val builder: VectorBuilder[Double] = new VectorBuilder(features.indices, features.values, features.indices.length, x.features.size)
                  builder.toSparseVector
                case _ => SparseVector(x.features.toArray)
              }
              features
            } else
              DenseVector(x.features.toArray))
      }

    val repartData =
      if (solverOptions.enableManualPartitionSize)
        objectifiedData.repartition(solverOptions.NUM_PART)
      else
        objectifiedData

    println("Running BinarySVMWithDBCFW solver")
    println(solverOptions)

    val (trainedModel, debugInfo) = new DBCFWSolverTuned[Vector[Double], Double](
      repartData,
      dissolveFunctions,
      solverOptions,
      miniBatchEnabled = false).optimize()

    // Dump debug information into a file
    val fw = new FileWriter(solverOptions.debugInfoPath)
    // Write the current parameters being used
    fw.write(solverOptions.toString())
    fw.write("\n")

    // Write spark-specific parameters
    fw.write(SolverUtils.getSparkConfString(data.context.getConf))
    fw.write("\n")

    // Write values noted from the run
    fw.write(debugInfo)
    fw.close()

    println(debugInfo)

    trainedModel

  }

}