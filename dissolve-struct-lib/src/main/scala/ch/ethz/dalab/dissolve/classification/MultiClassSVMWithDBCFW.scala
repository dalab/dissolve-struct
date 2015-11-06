package ch.ethz.dalab.dissolve.classification

import ch.ethz.dalab.dissolve.optimization.SolverOptions
import org.apache.spark.rdd.RDD
import java.io.FileWriter
import ch.ethz.dalab.dissolve.regression.LabeledObject
import org.apache.spark.mllib.regression.LabeledPoint
import breeze.linalg._
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.optimization.DBCFWSolverTuned

case class MultiClassLabel(label: Double, numClasses: Int)

object MultiClassSVMWithDBCFW extends DissolveFunctions[Vector[Double], MultiClassLabel] {

  /**
   * Feature function
   *
   * Analogous to phi(y) in (2)
   * Returns y_i * x_i
   *
   */
  def featureFn(x: Vector[Double], y: MultiClassLabel): Vector[Double] = {
    assert(y.label.toInt < y.numClasses,
      "numClasses = %d. Found y_i.label = %d"
        .format(y.numClasses, y.label.toInt))

    val featureVector = Vector.zeros[Double](x.size * y.numClasses)
    val numDims = x.size

    // Populate the featureVector in blocks [<class-0 features> <class-1 features> ...].
    val startIdx = y.label.toInt * numDims
    val endIdx = startIdx + numDims

    featureVector(startIdx until endIdx) := x

    featureVector
  }

  /**
   * Loss function
   *
   * Returns 0 if yTruth == yPredict, 1 otherwise
   * Equivalent to max(0, 1 - y w^T x)
   */
  def lossFn(yTruth: MultiClassLabel, yPredict: MultiClassLabel): Double =
    if (yTruth.label == yPredict.label)
      0.0
    else
      1.0

  /**
   * Maximization Oracle
   *
   * Want: argmax L(y_i, y) - <w, psi_i(y)>
   * This returns the most violating (Loss-augmented) label.
   */
  def oracleFn(model: StructSVMModel[Vector[Double], MultiClassLabel], xi: Vector[Double], yi: MultiClassLabel): MultiClassLabel = {

    val weights = model.getWeights()
    val numClasses = yi.numClasses

    // Obtain a list of scores for each class
    val mostViolatedContraint: (Double, Double) =
      (0 until numClasses).map {
        case cl =>
          (cl, weights dot featureFn(xi, MultiClassLabel(cl, numClasses)))
      }.map {
        case (cl, score) =>
          (cl.toDouble, score + 1.0)
      }.map { // Loss-augment the scores
        case (cl, score) =>
          if (yi.label == cl)
            (cl, score - 1.0)
          else
            (cl, score)
      }.maxBy { // Obtain the class with the maximum value
        case (cl, score) => score
      }

    MultiClassLabel(mostViolatedContraint._1, numClasses)
  }

  /**
   * Prediction function
   */
  def predictFn(model: StructSVMModel[Vector[Double], MultiClassLabel], xi: Vector[Double]): MultiClassLabel = {

    val weights = model.getWeights()
    val numClasses = model.numClasses

    assert(numClasses > 1)

    val prediction =
      (0 until numClasses).map {
        case cl =>
          (cl.toDouble, weights dot featureFn(xi, MultiClassLabel(cl, numClasses)))
      }.maxBy { // Obtain the class with the maximum value
        case (cl, score) => score
      }

    MultiClassLabel(prediction._1, numClasses)

  }

  /**
   * Classifying with in-built functions
   *
   * data needs to be 0-indexed
   */
  def train(
    data: RDD[LabeledPoint],
    numClasses: Int,
    solverOptions: SolverOptions[Vector[Double], MultiClassLabel]): StructSVMModel[Vector[Double], MultiClassLabel] = {

    solverOptions.numClasses = numClasses

    // Convert the RDD[LabeledPoint] to RDD[LabeledObject]
    val objectifiedData: RDD[LabeledObject[Vector[Double], MultiClassLabel]] =
      data.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], MultiClassLabel](MultiClassLabel(x.label, numClasses), SparseVector(x.features.toArray))
      }

    val repartData =
      if (solverOptions.enableManualPartitionSize)
        objectifiedData.repartition(solverOptions.NUM_PART)
      else
        objectifiedData

    println(solverOptions)

    val (trainedModel, debugInfo) = new DBCFWSolverTuned[Vector[Double], MultiClassLabel](
      repartData,
      this,
      solverOptions,
      miniBatchEnabled = false).optimize()

    println(debugInfo)

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

    trainedModel

  }

  /**
   * Classifying with user-submitted functions
   */
  def train(
    data: RDD[LabeledPoint],
    dissolveFunctions: DissolveFunctions[Vector[Double], MultiClassLabel],
    //featureFn: (Vector[Double], MultiClassLabel) => Vector[Double], // (y, x) => FeatureVector
    //lossFn: (MultiClassLabel, MultiClassLabel) => Double, // (yTruth, yPredict) => LossValue
    //oracleFn: (StructSVMModel[Vector[Double], MultiClassLabel], Vector[Double], MultiClassLabel) => MultiClassLabel, // (model, y_i, x_i) => Label
    //predictFn: (StructSVMModel[Vector[Double], MultiClassLabel], Vector[Double]) => MultiClassLabel,
    solverOptions: SolverOptions[Vector[Double], MultiClassLabel]): StructSVMModel[Vector[Double], MultiClassLabel] = {

    val numClasses = solverOptions.numClasses
    assert(numClasses > 1)

    val minlabel = data.map(_.label).min()
    val maxlabel = data.map(_.label).max()
    assert(minlabel == 0, "Label classes need to be 0-indexed")
    assert(maxlabel - minlabel + 1 == numClasses,
      "Number of classes in data do not tally with passed argument")

    // Convert the RDD[LabeledPoint] to RDD[LabeledObject]
    val objectifiedData: RDD[LabeledObject[Vector[Double], MultiClassLabel]] =
      data.map {
        case x: LabeledPoint =>
          new LabeledObject[Vector[Double], MultiClassLabel](MultiClassLabel(x.label, numClasses),
            if (solverOptions.sparse)
              SparseVector(x.features.toArray)
            else
              Vector(x.features.toArray))
      }

    val repartData =
      if (solverOptions.enableManualPartitionSize)
        objectifiedData.repartition(solverOptions.NUM_PART)
      else
        objectifiedData

    println(solverOptions)

    val (trainedModel, debugInfo) = new DBCFWSolverTuned[Vector[Double], MultiClassLabel](
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