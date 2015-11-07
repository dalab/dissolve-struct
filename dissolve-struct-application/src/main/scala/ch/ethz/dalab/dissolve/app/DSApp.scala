package ch.ethz.dalab.dissolve.app

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import breeze.linalg.Vector
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions
import ch.ethz.dalab.dissolve.optimization.GapThresholdCriterion
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.regression.LabeledObject

/**
 * This defines the x-part of the training example
 * For example, in case of sequence OCR this would be a (d x n) matrix, with
 * each column containing the pixel representation of a character
 */
case class Pattern() {

}

/**
 * This defined the y-part of the training example
 * Once again, in case of OCR this would be a n-dimensional vector, with the
 * i-th element containing label for i-th character of x.
 */
case class Label() {

}

/**
 * This is the core of your Structured SVM application.
 * In here, you'll find three functions and a driver program that you'll need
 * to fill in to get your application running.
 *
 * The interface is inspired by SVM^struct by Joachims et al.
 * (http://www.cs.cornell.edu/people/tj/svm_light/svm_struct.html)
 */
object DSApp extends DissolveFunctions[Pattern, Label] {

  /**
   * ============== Joint feature Map: \phi(x, y) ==============
   *
   * This encodes the complex input-output (x, y) pair in a Vector space (done
   * using vectors from the Breeze library)
   */
  def featureFn(x: Pattern, y: Label): Vector[Double] = {

    // Insert code here for Joint feature map here

    ???
  }

  /**
   * ============== Structured Loss Function: \Delta(y, y^m) ==============
   *
   * Loss for predicting <yPredicted> instead of <yTruth>.
   * This needs to be 0 if <yPredicted> == <yTruth>
   */
  def lossFn(yPredicted: Label, yTruth: Label): Double = {

    // Insert code for Loss function here

    ???
  }

  /**
   * ============== Maximization Oracle: H^m(w) ==============
   *
   * Finds the most violating constraint by solving the loss-augmented decoding
   * subproblem.
   * This is equivalent to predicting
   * y* = argmax_{y} \Delta(y, y^m) + < w, \phi(x^m, y) >
   * for some training example (x^m, y^m) and parameters w
   *
   * Make sure the loss-augmentation is consistent with the \Delta defined above.
   *
   * By default, the prediction function calls this oracle with y^m = null.
   * In which case, the loss-augmentation can be skipped using a simple check
   * on y^m.
   *
   * For examples, or common oracle/decoding functions (like BP Loopy, Viterbi
   * or BP on Chain CF) refer to the examples package.
   */
  def oracleFn(model: StructSVMModel[Pattern, Label], x: Pattern, y: Label): Label = {

    val weightVec = model.weights

    // Insert code for maximization Oracle here

    ???
  }

  /**
   * ============== Prediction Function ==============
   *
   * Finds the best output candidate for x, given parameters w.
   * This is equivalent to solving:
   * y* = argmax_{y} < w, \phi(x^m, y) >
   *
   * Note that this is very similar to the maximization oracle, but without
   * the loss-augmentation. So, by default, we call the oracle function by
   * setting y as null.
   */
  def predictFn(model: StructSVMModel[Pattern, Label], x: Pattern): Label =
    oracleFn(model, x, null)

  /**
   * ============== Driver ==============
   *
   * This is the entry point into the program.
   * In here, we initialize the SparkContext, set the parameters and call the
   * optimization routine.
   *
   * To begin with the training, we'll need three things:
   * a. A SparkContext instance (Defaults provided)
   * b. Solver Parameters (Defaults provided)
   * c. Data
   *
   * To execute, you should package this into a jar and provide it using
   * spark-submit (http://spark.apache.org/docs/latest/submitting-applications.html).
   *
   * Alternately, you can right-click and Run As -> Scala Application to run
   * within Eclipse.
   */
  def main(args: Array[String]): Unit = {

    val appname = "DSApp"

    /**
     * ============== Initialize Spark ==============
     *
     * Alternately, use:
     * val conf = new SparkConf().setAppName(appname).setMaster("local[4]")
     * if you're planning to execute within Eclipse using 4 cores
     */
    val conf = new SparkConf().setAppName(appname)
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    /**
     * ============== Set Solver parameters ==============
     */
    val solverOptions = new SolverOptions[Pattern, Label]()
    // Regularization paramater
    solverOptions.lambda = 0.01

    // Stopping criterion
    solverOptions.stoppingCriterion = GapThresholdCriterion
    solverOptions.gapThreshold = 1e-3
    solverOptions.gapCheck = 25 // Checks for gap every gapCheck rounds

    // Set the fraction of data to be used in training during each round
    // In this case, 50% of the data is uniformly sampled for training at the
    // beginning of each round
    solverOptions.sampleFrac = 0.5

    // Set how many partitions you want to split the data into.
    // These partitions will be local to each machine and the respective dual
    // variables associated with these partitions will reside locally.
    // Ideally, you want to set this to: #cores x #workers x 2.
    // If this is disabled, Spark decides on the partitioning, which be may
    // be suboptimal.
    solverOptions.enableManualPartitionSize = true
    solverOptions.NUM_PART = 8

    // Optionally, you can enable obtaining additional statistics like the
    // the training, test errors w.r.t to rounds, along with the gap
    // This is expensive as it involves a complete pass through the data.
    solverOptions.debug = false
    // This computes the statistics every debugMultiplier^i rounds.
    // So, in this case, it does so in 1, 2, 4, 8, ...
    // Beyond the 50th round, statistics is collected every 10 rounds.
    solverOptions.debugMultiplier = 2
    // Writes the statistics in CSV format in the provided path
    solverOptions.debugInfoPath = "path/to/statistics.csv"

    /**
     * ============== Provide Data ==============
     */
    val trainDataRDD: RDD[LabeledObject[Pattern, Label]] = {

      // Insert code to load TRAIN data here

      ???
    }
    val testDataRDD: RDD[LabeledObject[Pattern, Label]] = {

      // Insert code to load TEST data here

      ???
    }
    // Optionally, set to None in case you don't want statistics on test data
    solverOptions.testDataRDD = Some(testDataRDD)

    /**
     * ============== Training ==============
     */
    val trainer: StructSVMWithDBCFW[Pattern, Label] =
      new StructSVMWithDBCFW[Pattern, Label](
        trainDataRDD,
        DSApp,
        solverOptions)

    val model: StructSVMModel[Pattern, Label] = trainer.trainModel()

    /**
     * ============== Store Model ==============
     *
     * Optionally, you can store the model's weight parameters.
     *
     * To load a model, you can use
     * val weights = breeze.linalg.csvread(new java.io.File(weightOutPath))
     * val model = new StructSVMModel[Pattern, Label](weights, 0.0, null, DSApp)
     */
    val weightOutPath = "path/to/weights.csv"
    val weights = model.weights.toDenseVector.toDenseMatrix
    breeze.linalg.csvwrite(new java.io.File(weightOutPath), weights)

  }
}