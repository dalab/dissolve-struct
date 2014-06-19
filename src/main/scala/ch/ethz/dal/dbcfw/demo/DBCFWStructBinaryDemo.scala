/**
 *
 */
package ch.ethz.dal.dbcfw.demo

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg._
import breeze.numerics._
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import ch.ethz.dal.dbcfw.classification.StructSVMWithSSG
import ch.ethz.dal.dbcfw.classification.StructSVMWithSSG
import ch.ethz.dal.dbcfw.classification.StructSVMModel

/**
 *
 */
object DBCFWStructBinaryDemo {

  /**
   * Feature function
   *
   * Analogous to phi(y) in (2)
   * Returns y_i * x_i
   */
  def featureFnBin(y: Double, x: Vector[Double]): Vector[Double] =
    x :* y;

  /**
   * Represent feature vector as 1xn Matrix
   */
  def featureFn(y: Vector[Double], x: Matrix[Double]): Vector[Double] =
    x.toDenseMatrix.toDenseVector :* y(0)

  /**
   * Loss function
   *
   * Returns 0 if yTruth == yPredict, 1 otherwise
   * Equivalent to max(0, 1 - y w^T x)
   */
  def lossFnBin(yTruth: Double, yPredict: Double): Double =
    abs(yTruth - yPredict)

  def lossFn(yTruth: Vector[Double], yPredict: Vector[Double]): Double =
    abs(yTruth(0) - yPredict(0))

  /**
   * Maximization Oracle
   *
   * Want: max L(y_i, y) - <w, psi_i(y)>
   * This returns a negative value in case of a correct prediction and positive in case of a false prediction
   */
  def oracleFnBin(weights: Vector[Double], yi: Double, xi: Vector[Double]): Double =
    lossFnBin(yi, weights dot xi) - (weights dot featureFnBin(yi, xi))

  def oracleFn(model: StructSVMModel, yi: Vector[Double], xi: Matrix[Double]): Vector[Double] =
    DenseVector.fill(1) { lossFn(yi, DenseVector.fill(1) { model.getWeights().toDenseVector dot xi.toDenseMatrix.toDenseVector }) - (model.getWeights() dot featureFn(yi, xi)) }

  /**
   * Prediction function
   */
  def predictFnBin(weights: Vector[Double], xi: Vector[Double]): Double =
    if ((weights dot featureFnBin(1.0, xi)) > (weights dot featureFnBin(-1.0, xi)))
      1.0
    else
      -1.0

  def predictFn(model: StructSVMModel, xi: Matrix[Double]): Vector[Double] =
    if ((model.getWeights() dot featureFn(DenseVector.ones[Double](1), xi)) > (model.getWeights() dot featureFn(DenseVector.fill[Double](1) { -1.0 }, xi)))
      DenseVector.ones[Double](1)
    else
      DenseVector.fill[Double](1) { -1.0 }

  def loadLibSVMFile(filename: String): Vector[LabeledObject] = {
    var n: Int = 0
    var ndims: Int = 0

    // First pass, get number of data points and number of features
    for (line ← scala.io.Source.fromFile(filename).getLines()) {
      n += 1
      ndims = max(ndims, max(line.split(" ").slice(1, line.split(" ").size).map(s ⇒ s.split(":")(0) toInt)))
    }

    // Create a n-dim vector, each element containing a 1-dim vector
    // val labels: DenseVector[DenseVector[Double]] = DenseVector.fill(n) { DenseVector.zeros(1) }
    // Create a n-dim vector, each element containing a 1xndim dimensional matrix
    // val patterns: DenseVector[DenseMatrix[Double]] = DenseVector.fill(n) { DenseMatrix.zeros(1, ndims) }
    // Create LabeledObject s

    // Second pass - Fill in data
    val data: Vector[LabeledObject] = DenseVector.fill(n) { new LabeledObject(DenseVector.zeros(1), DenseMatrix.zeros(1, ndims)) }
    var row: Int = 0
    for (line ← scala.io.Source.fromFile(filename).getLines()) {
      val content: Array[String] = line.split(" ")
      data(row).label(0) = content(0) toInt

      for (ele ← content.slice(1, content.size)) {
        val col = ele.split(":")(0) toInt
        val item = ele.split(":")(1) toDouble

        data(row).pattern(1, col) = item
      }
      row += 1
    }

    data
  }

  def main(args: Array[String]): Unit = {
    val data: Vector[LabeledObject] = loadLibSVMFile("data/a1a.txt")

    val trainer: StructSVMWithSSG = new StructSVMWithSSG(data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn)
    
    val model:StructSVMModel = trainer.trainModel()

  }

}