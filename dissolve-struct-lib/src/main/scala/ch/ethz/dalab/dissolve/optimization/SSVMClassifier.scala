package ch.ethz.dalab.dissolve.optimization

import java.io.File

import scala.reflect.ClassTag

import org.apache.spark.rdd.RDD

import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.regression.LabeledObject

class SSVMClassifier[X, Y](protected val model: DissolveFunctions[X, Y]) {

  protected var weights: Vector[Double] = Vector.zeros(0) // Begin with an empty vector

  /**
   * Distributed Optimization
   */
  def train(trainData: RDD[LabeledObject[X, Y]],
            testData: Option[RDD[LabeledObject[X, Y]]],
            solver: DistributedSolver[X, Y])(implicit m: ClassTag[Y]): Unit = {
    weights = solver.train(trainData, testData)
  }

  def train(trainData: RDD[LabeledObject[X, Y]],
            testData: RDD[LabeledObject[X, Y]],
            solver: DistributedSolver[X, Y])(implicit m: ClassTag[Y]): Unit =
    train(trainData, Some(testData), solver)

  def train(trainData: RDD[LabeledObject[X, Y]],
            solver: DistributedSolver[X, Y])(implicit m: ClassTag[Y]): Unit =
    train(trainData, None, solver)

  /**
   * Local Optimization
   */
  def train(trainData: Seq[LabeledObject[X, Y]],
            testData: Option[Seq[LabeledObject[X, Y]]],
            solver: LocalSolver[X, Y])(implicit m: ClassTag[Y]): Unit = {
    weights = solver.train(trainData, testData)
  }

  def train(trainData: Seq[LabeledObject[X, Y]],
            testData: Seq[LabeledObject[X, Y]],
            solver: LocalSolver[X, Y])(implicit m: ClassTag[Y]): Unit =
    train(trainData, Some(testData), solver)

  def train(trainData: Seq[LabeledObject[X, Y]],
            solver: LocalSolver[X, Y])(implicit m: ClassTag[Y]): Unit =
    train(trainData, None, solver)

  /**
   * Prediction
   */
  def predict(x: X): Y =
    model.predictFn(weights, x)

  /**
   * Weight vector related operations
   */

  def getWeights(): Vector[Double] = weights.copy

  def setWeights(weights: Vector[Double]) = { this.weights = weights }

  def saveWeights(filepath: String) = { breeze.linalg.csvwrite(new File(filepath), weights.toDenseVector.toDenseMatrix) }

  def loadWeights(filepath: String) = { weights = breeze.linalg.csvread(new File(filepath)).toDenseVector.toVector }

}