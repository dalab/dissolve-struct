package ch.ethz.dalab.dissolve.classification

import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.regression.LabeledObject
import org.apache.spark.mllib.regression.LabeledPoint
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.optimization.BCFWSolver
import ch.ethz.dalab.dissolve.regression.LabeledObject
import java.io.FileWriter
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg.SparseVector

object MultiClassSVMWithBCFW extends DissolveFunctions[Vector[Double], MultiClassLabel] {
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
  override def oracleFn(model: StructSVMModel[Vector[Double], MultiClassLabel], xi: Vector[Double], yi: MultiClassLabel): MultiClassLabel = {

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
   * Classifying with user-submitted functions
   */
  def train(
    data: Seq[LabeledPoint],
    dissolveFunctions: DissolveFunctions[Vector[Double], MultiClassLabel],
    solverOptions: SolverOptions[Vector[Double], MultiClassLabel]): StructSVMModel[Vector[Double], MultiClassLabel] = {

    val numClasses = solverOptions.numClasses
    assert(numClasses > 1)

    val objectifiedData: Seq[LabeledObject[Vector[Double], MultiClassLabel]] = data.map {
      case x: LabeledPoint =>
        new LabeledObject[Vector[Double], MultiClassLabel](MultiClassLabel(x.label, numClasses),
          if (solverOptions.sparse)
            SparseVector(x.features.toArray)
          else
            Vector(x.features.toArray))
    }

    println(solverOptions)

    val (trainedModel, debugInfo) = new BCFWSolver[Vector[Double], MultiClassLabel](
      objectifiedData,
      dissolveFunctions,
      solverOptions).optimize()

    // Dump debug information into a file
    val fw = new FileWriter(solverOptions.debugInfoPath)
    // Write the current parameters being used
    fw.write(solverOptions.toString())
    fw.write("\n")

    // Write values noted from the run
    fw.write(debugInfo)
    fw.close()

    println(debugInfo)

    trainedModel
  }

}