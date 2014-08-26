package ch.ethz.dal.dbcfw.demo.chain

import ch.ethz.dal.dbcfw.regression.LabeledObject
import breeze.linalg._
import breeze.numerics._
import breeze.generic._
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import ch.ethz.dal.dbcfw.classification.StructSVMWithSSG
import java.io.File
import ch.ethz.dal.dbcfw.optimization.SolverOptions
import ch.ethz.dal.dbcfw.classification.StructSVMWithBCFW
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import ch.ethz.dal.dbcfw.optimization.DBCFWSolver
import java.io.PrintWriter
import ch.ethz.dal.dbcfw.optimization.SolverUtils
import org.apache.log4j.Logger
import collection.JavaConversions.enumerationAsScalaIterator
import org.apache.log4j.PropertyConfigurator
import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.classification.Types._

/**
 * LogHelper is a trait you can mix in to provide easy log4j logging
 * for your scala classes.
 */
trait LogHelper {
  val loggerName = this.getClass.getName
  lazy val logger = Logger.getLogger(loggerName)
}

object ChainDemo extends LogHelper {

  val debugOn = true

  /**
   * Reads data produced by the convert-ocr-data.py script and loads into memory as a vector of Labeled objects
   *
   *  TODO
   *  * Take foldNumber as a parameter and return training and test set
   */
  def loadData(patternsFilename: String, labelsFilename: String, foldFilename: String): DenseVector[LabeledObject] = {
    val patterns: Array[String] = scala.io.Source.fromFile(patternsFilename).getLines().toArray[String]
    val labels: Array[String] = scala.io.Source.fromFile(labelsFilename).getLines().toArray[String]
    val folds: Array[String] = scala.io.Source.fromFile(foldFilename).getLines().toArray[String]

    val n = labels.size

    assert(patterns.size == labels.size, "#Patterns=%d, but #Labels=%d".format(patterns.size, labels.size))
    assert(patterns.size == folds.size, "#Patterns=%d, but #Folds=%d".format(patterns.size, folds.size))

    val data: DenseVector[LabeledObject] = DenseVector.fill(n) { null }

    for (i <- 0 until n) {
      // Expected format: id, #rows, #cols, (pixels_i_j,)* pixels_n_m
      val patLine: List[Double] = patterns(i).split(",").map(x => x.toDouble) toList
      // Expected format: id, #letters, (letters_i)* letters_n
      val labLine: List[Double] = labels(i).split(",").map(x => x.toDouble) toList

      val patNumRows: Int = patLine(1) toInt
      val patNumCols: Int = patLine(2) toInt
      val labNumEles: Int = labLine(1) toInt

      assert(patNumCols == labNumEles, "pattern_i.cols == label_i.cols violated in data")

      val patVals: Array[Double] = patLine.slice(3, patLine.size).toArray[Double]
      // The pixel values should be Column-major ordered
      val thisPattern: DenseMatrix[Double] = DenseVector(patVals).toDenseMatrix.reshape(patNumRows, patNumCols)

      val labVals: Array[Double] = labLine.slice(2, labLine.size).toArray[Double]
      assert(List.fromArray(labVals).count(x => x < 0 || x > 26) == 0, "Elements in Labels should be in the range [0, 25]")
      val thisLabel: DenseVector[Double] = DenseVector(labVals)

      assert(thisPattern.cols == thisLabel.size, "pattern_i.cols == label_i.cols violated in Matrix representation")

      data(i) = new LabeledObject(thisLabel, thisPattern)

    }

    data
  }

  /**
   * Returns a vector, capturing unary, bias and pairwise features of the word
   *
   * x is a Pattern matrix, of dimensions numDims x NumVars (say 129 x 9)
   * y is a Label vector, of dimension numVars (9 for above x)
   */
  def featureFn(y: Vector[Double], xM: Matrix[Double]): Vector[Double] = {
    val x = xM.toDenseMatrix
    val numStates = 26
    val numDims = x.rows // 129 in case of Chain OCR
    val numVars = x.cols
    // First term for unaries, Second term for first and last letter baises, Third term for Pairwise features
    // Unaries are row-major ordered, i.e., [0,129) positions for 'a', [129, 258) for 'b' and so on 
    val phi: DenseVector[Double] = DenseVector.zeros[Double]((numStates * numDims) + (2 * numStates) + (numStates * numStates))

    /* Unaries */
    for (i <- 0 until numVars) {
      val idx = (y(i).toInt * numDims)
      phi((idx) until (idx + numDims)) :=
        phi((idx) until (idx + numDims)) + x(::, i)
    }

    phi(numStates * numDims + y(0).toInt) = 1.0
    phi(numStates * numDims + numStates + y(-1).toInt) = 1.0

    /* Pairwise */
    val offset = (numStates * numDims) + (2 * numStates)
    for (i <- 0 until (numVars - 1)) {
      val idx = y(i).toInt + numStates * y(i + 1).toInt
      phi(offset + idx) = phi(offset + idx) + 1.0
    }

    phi
  }

  /**
   * Return Normalized Hamming distance
   */
  def lossFn(yTruth: Vector[Double], yPredict: Vector[Double]): Double =
    sum((yTruth :== yPredict).map(x => if (x) 0 else 1)) / yTruth.size.toDouble

  /**
   * Readable representation of the weight vector
   */
  class Weight(
    val unary: DenseMatrix[Double],
    val firstBias: DenseVector[Double],
    val lastBias: DenseVector[Double],
    val pairwise: DenseMatrix[Double]) {
  }

  /**
   * Converts weight vector into Weight object, by partitioning it into unary, bias and pairwise terms
   */
  def weightVecToObj(weightVec: Vector[Double], numStates: Int, numDims: Int): Weight = {
    val idx: Int = numStates * numDims

    val unary: DenseMatrix[Double] = weightVec(0 until idx)
      .toDenseVector
      .toDenseMatrix
      .reshape(numDims, numStates)
    val firstBias: Vector[Double] = weightVec(idx until (idx + numStates))
    val lastBias: Vector[Double] = weightVec((idx + numStates) until (idx + 2 * numStates))
    val pairwise: DenseMatrix[Double] = weightVec((idx + 2 * numStates) until weightVec.size)
      .toDenseVector
      .toDenseMatrix
      .reshape(numStates, numStates)

    new Weight(unary, firstBias.toDenseVector, lastBias.toDenseVector, pairwise)
  }

  /**
   * Replicate the functionality of Matlab's max function
   * Returns 2-row vectors in the form a Matrix
   * 1st row contains column-wise max of the Matrix
   * 2nd row contains corresponding indices of the max elements
   */
  def columnwiseMax(matM: Matrix[Double]): DenseMatrix[Double] = {
    val mat: DenseMatrix[Double] = matM.toDenseMatrix
    val colMax: DenseMatrix[Double] = DenseMatrix.zeros[Double](2, mat.cols)

    for (col <- 0 until mat.cols) {
      // 1st row contains max
      colMax(0, col) = max(mat(::, col))
      // 2nd row contains indices of the max
      colMax(1, col) = argmax(mat(::, col))
    }
    colMax
  }

  /**
   * Log decode, with forward and backward passes
   * Was ist das?
   */
  def logDecode(logNodePotMat: Matrix[Double], logEdgePotMat: Matrix[Double]): Vector[Double] = {

    val logNodePot: DenseMatrix[Double] = logNodePotMat.toDenseMatrix
    val logEdgePot: DenseMatrix[Double] = logEdgePotMat.toDenseMatrix

    val nNodes: Int = logNodePot.rows
    val nStates: Int = logNodePot.cols

    /*--- Forward pass ---*/
    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](nNodes, nStates) // nx26 matrix
    val mxState: DenseMatrix[Double] = DenseMatrix.zeros[Double](nNodes, nStates) // nx26 matrix
    alpha(0, ::) := logNodePot(0, ::)
    for (n <- 1 until nNodes) {
      /* Equivalent to `tmp = repmat(alpha(n-1, :)', 1, nStates) + logEdgePot` */
      // Create an empty 26x26 repmat term
      val alphaRepmat: DenseMatrix[Double] = DenseMatrix.zeros[Double](nStates, nStates)
      for (col <- 0 until nStates) {
        // Take the (n-1)th row from alpha and represent it as a column in repMat
        // alpha(n-1, ::) returns a Transposed view, so use the below workaround
        alphaRepmat(::, col) := alpha.t(::, n - 1)
      }
      val tmp: DenseMatrix[Double] = alphaRepmat + logEdgePot
      val colMaxTmp: DenseMatrix[Double] = columnwiseMax(tmp)
      alpha(n, ::) := logNodePot(n, ::) + colMaxTmp(0, ::)
      mxState(n, ::) := colMaxTmp(1, ::)
    }
    /*--- Backward pass ---*/
    val y: DenseVector[Double] = DenseVector.zeros[Double](nNodes)
    // [dummy, y(nNodes)] = max(alpha(nNodes, :))
    y(nNodes - 1) = argmax(alpha.t(::, nNodes - 1).toDenseVector)
    for (n <- nNodes - 2 to 0 by -1) {
      y(n) = mxState(n + 1, y(n + 1).toInt)
    }
    y
  }

  /**
   * The Maximization Oracle
   */
  def oracleFn(model: StructSVMModel, yi: Vector[Double], xi: Matrix[Double]): Vector[Double] = {
    val numStates = 26
    // val xi = xiM.toDenseMatrix // 129 x n matrix, ex. 129 x 9 if len(word) = 9
    val numDims = xi.rows // 129 in Chain example 
    val numVars = xi.cols // The length of word, say 9

    // Convert the lengthy weight vector into an object, to ease representation
    // weight.unary is a numDims x numStates Matrix (129 x 26 in above example)
    // weight.firstBias and weight.lastBias is a numStates-dimensional vector
    // weight.pairwise is a numStates x numStates Matrix
    val weight: Weight = weightVecToObj(model.getWeights(), numStates, numDims)

    val thetaUnary: DenseMatrix[Double] = weight.unary.t * xi // Produces a 26 x n matrix

    // First position has a bias
    thetaUnary(::, 0) := thetaUnary(::, 0) + weight.firstBias
    // Last position has a bias
    thetaUnary(::, -1) := thetaUnary(::, -1) + weight.lastBias

    val thetaPairwise: DenseMatrix[Double] = weight.pairwise

    // Add loss-augmentation to the score (normalized Hamming distances used for loss)
    val l: Int = yi.size
    for (i <- 0 until numVars) {
      thetaUnary(::, i) := thetaUnary(::, i) + 1.0 / l
      val idx = yi(i).toInt
      thetaUnary(idx, i) = thetaUnary(idx, i) - 1.0 / l
    }

    // Solve inference problem
    val label: Vector[Double] = logDecode(thetaUnary.t, thetaPairwise) // - 1.0

    label
  }

  /**
   * Loss function
   *
   * TODO
   * * Use MaxOracle instead of this (Use yi: Option<Vector[Double]>)
   */
  def predictFn(model: StructSVMModel, xi: Matrix[Double]): Vector[Double] = {
    val numStates = 26
    // val xi = xiM.toDenseMatrix // 129 x n matrix, ex. 129 x 9 if len(word) = 9
    val numDims = xi.rows // 129 in Chain example 
    val numVars = xi.cols // The length of word, say 9

    // Convert the lengthy weight vector into an object, to ease representation
    // weight.unary is a numDims x numStates Matrix (129 x 26 in above example)
    // weight.firstBias and weight.lastBias is a numStates-dimensional vector
    // weight.pairwise is a numStates x numStates Matrix
    val weight: Weight = weightVecToObj(model.getWeights(), numStates, numDims)

    val thetaUnary: DenseMatrix[Double] = weight.unary.t * xi // Produces a 26 x n matrix

    // First position has a bias
    thetaUnary(::, 0) := thetaUnary(::, 0) + weight.firstBias
    // Last position has a bias
    thetaUnary(::, -1) := thetaUnary(::, -1) + weight.lastBias

    val thetaPairwise: DenseMatrix[Double] = weight.pairwise

    // Solve inference problem
    val label: Vector[Double] = logDecode(thetaUnary.t, thetaPairwise) // - 1.0

    label
  }

  /**
   * Convert Vector[Double] to respective String representation
   */
  def labelVectorToString(vec: Vector[Double]): String =
    if (vec.size == 0)
      ""
    else if (vec.size == 1)
      (vec(0).toInt + 97).toChar + ""
    else
      (vec(0).toInt + 97).toChar + labelVectorToString(vec(1 until vec.size))

  /**
   * ****************************************************************
   *    ___   _____ ____ _      __
   *   / _ ) / ___// __/| | /| / /
   *  / _  |/ /__ / _/  | |/ |/ /
   * /____/ \___//_/    |__/|__/
   *
   * (Averaging of primal variables after each round)
   * ****************************************************************
   */
  def chainBCFW(): Unit = {

    val PERC_TRAIN: Double = 0.1 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val train_data_unord: Vector[LabeledObject] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val test_data: Vector[LabeledObject] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    // val train_data = train_data_unord(List.fromArray(perm))
    val train_data: DenseVector[LabeledObject] = train_data_unord(List.fromArray(perm).slice(0, (PERC_TRAIN * train_data_unord.size).toInt)).toDenseVector

    if (debugOn) {
      println("Loaded %d training examples, pattern:%dx%d and labels:%dx1"
        .format(train_data.size,
          train_data(0).pattern.rows,
          train_data(0).pattern.cols,
          train_data(0).label.size))
      println("Loaded %d test examples, pattern:%dx%d and labels:%dx1"
        .format(test_data.size,
          test_data(0).pattern.rows,
          test_data(0).pattern.cols,
          test_data(0).label.size))
    }

    val solverOptions: SolverOptions = new SolverOptions();
    solverOptions.numPasses = 5
    solverOptions.debug = true
    solverOptions.xldebug = false
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = true
    solverOptions.testData = test_data

    /*val trainer: StructSVMWithSSG = new StructSVMWithSSG(train_data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)*/

    val trainer: StructSVMWithBCFW = new StructSVMWithBCFW(train_data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)

    val model: StructSVMModel = trainer.trainModel()

    var avgTrainLoss: Double = 0.0
    for (item <- train_data) {
      val prediction = model.predictFn(model, item.pattern)
      avgTrainLoss += lossFn(item.label, prediction)
      // if (debugOn)
      // println("Truth = %-10s\tPrediction = %-10s".format(labelVectorToString(item.label), labelVectorToString(prediction)))
    }
    println("Average loss on training set = %f".format(avgTrainLoss / train_data.size))

    var avgTestLoss: Double = 0.0
    for (item <- test_data) {
      val prediction = model.predictFn(model, item.pattern)
      avgTestLoss += lossFn(item.label, prediction)
      // if (debugOn)
      // println("Truth = %-10s\tPrediction = %-10s".format(labelVectorToString(item.label), labelVectorToString(prediction)))
    }
    println("Average loss on test set = %f".format(avgTestLoss / test_data.size))

  }

  /**
   * ****************************************************************
   *    ___        ___   _____ ____ _      __
   *   / _ \ ____ / _ ) / ___// __/| | /| / /
   *  / // //___// _  |/ /__ / _/  | |/ |/ /
   * /____/     /____/ \___//_/    |__/|__/
   *
   * (With averaging of primal variables)
   * ****************************************************************
   */
  def chainDBCFWwAvg(): Unit = { //TODO: delete (is a solver option already instead)

    val NUM_ROUNDS: Int = 50
    val NUM_PART: Int = 2
    val PERC_TRAIN: Double = 0.1 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val trainDataUnord: Vector[LabeledObject] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val testDataUnord: Vector[LabeledObject] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    val conf = new SparkConf().setAppName("Chain-DBCFW").setMaster("local")
    val sc = new SparkContext(conf)

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    // val train_data = trainDataUnord(List.fromArray(perm))
    val train_data: DenseVector[LabeledObject] = trainDataUnord(List.fromArray(perm).slice(0, (PERC_TRAIN * trainDataUnord.size).toInt)).toDenseVector

    val train_rdd: RDD[LabeledObject] = sc.parallelize(train_data.toArray, 2)

    val solverOptions: SolverOptions = new SolverOptions();
    solverOptions.numPasses = 1 // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = false
    solverOptions.xldebug = false
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = false

    val d: Int = featureFn(train_rdd.first.label, train_rdd.first.pattern).size
    // Let the initial model contain zeros for all weights
    val model: StructSVMModel = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    val startTime = System.currentTimeMillis()
    logger.info("[DATA] round,time,train_error,test_error")

    for (roundNum <- 1 to NUM_ROUNDS) {
      /**
       * Map step. Each partition of the training data produces an individual model
       * TODO Maybe convert more efficiently using an aggregate?
       */
      val trainedModels = train_rdd.mapPartitions(trainDataPartition => DBCFWSolver.optimize(trainDataPartition,
        model, featureFn, lossFn, oracleFn,
        predictFn, solverOptions, false), true)
      /**
       * Reduce step. Combine models into a single model.
       */
      val nextModel = DBCFWSolver.combineModels(trainedModels)
      /**
       * Communication step. Communicate the new model to all nodes. Resume training.
       */
      model.updateWeights(nextModel.getWeights)
      model.updateEll(nextModel.getEll)

      val trainError = SolverUtils.averageLoss(train_data, lossFn, predictFn, model)
      val testError = SolverUtils.averageLoss(testDataUnord, lossFn, predictFn, model)

      val elapsedTime = (System.currentTimeMillis() - startTime).toDouble / 1000.0

      logger.info("[DATA] %d,%f,%f,%f\n".format(roundNum, elapsedTime, trainError, testError))
      println("[Round #%d] Train loss = %f, Test loss = %f\n".format(roundNum, trainError, testError))

      /**
       * Shuffle maybe?
       */
    }

  }

  /**
   * ****************************************************************
   *    ___        ___   _____ ____ _      __
   *   / _ \ ____ / _ ) / ___// __/| | /| / /
   *  / // //___// _  |/ /__ / _/  | |/ |/ /
   * /____/     /____/ \___//_/    |__/|__/
   *
   * (Mini-Batch) (try)
   * ****************************************************************
   */
  def chainDBCFWminiBatch(): Unit = { //TODO: delete (is a solver option already instead)
    val NUM_ROUNDS: Int = 5
    val NUM_PART: Int = 2
    val PERC_TRAIN: Double = 0.1 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val trainDataUnord: Vector[LabeledObject] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val testDataUnord: Vector[LabeledObject] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    val conf = new SparkConf().setAppName("Chain-DBCFW").setMaster("local").set("spark.executor.memory", "1g")
    val sc = new SparkContext(conf)
    sc.setLocalProperty("spark.executor.memory", "1g")

    println(conf.get("spark.executor.memory"))
    println(sc.getLocalProperty("spark.executor.memory"))

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    val train_data: DenseVector[LabeledObject] = trainDataUnord(List.fromArray(perm).slice(0, (PERC_TRAIN * trainDataUnord.size).toInt)).toDenseVector

    val solverOptions: SolverOptions = new SolverOptions()
    solverOptions.numPasses = 1 // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = false
    solverOptions.xldebug = false
    solverOptions.lambda = 0.1
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = false

    logger.info("[DATA] lambda = " + solverOptions.lambda)

    val d: Int = featureFn(trainDataUnord(0).label, trainDataUnord(0).pattern).size
    // Let the initial model contain zeros for all weights
    val model: StructSVMModel = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    // Create a list containing LabeledObjects zipped with their w_i's and l_i's
    // val zippedTrainData: List[(LabeledObject, DenseVector[Double], Double)] = for (i <- (0 until train_data.size).toList) yield (train_data(i), DenseVector.zeros[Double](d), 0.0)
    var train_rdd: RDD[(LabeledObject, (DenseVector[Double], Double))] = sc.parallelize(for (i <- (0 until train_data.size).toArray) yield (train_data(i), (DenseVector.zeros[Double](d), 0.0)))
    // train_rdd.cache()

    logger.info("[DATA] round,train_error,test_error")
    // train_rdd.cache()

    for (roundNum <- 1 to NUM_ROUNDS) {
      /**
       * Map step. Each partition of the training data produces a model.
       * But, the model's weights only reflects changes in w's
       */
      val trainedZippedData: RDD[(LabeledObject, (DenseVector[Double], Double))] = train_rdd.mapPartitions(trainDataPartition => DBCFWSolver.bcfwOptimizeMiniBatch(trainDataPartition,
        model, featureFn, lossFn, oracleFn,
        predictFn, solverOptions), true)
      /**
       * Reduce step. Combine train_rdd and trainedZippedData into a new stream
       */
      // val newModel = DBCFWSolver.bcfwCombine(model, trainedZippedData)
      /*val newDiffs: (Vector[Double], Double) = train_rdd.sample(false, 0.1, 42)
        .mapPartitions(trainDataPartition => DBCFWSolver.bcfwOptimize(trainDataPartition, model, featureFn, lossFn, oracleFn, predictFn, solverOptions), true)
        .map(x => (x._2.toDenseVector, x._3)) // Map each element in RDD to (\Delta w_i, \Delta ell_i)
        .reduce((mA, mB) => (mA._1 + mB._1, mA._2 + mB._2)) //  Sum all \Delta w_i's and \Delta ell_i's
      */

      // Now, train_rdd contains w_i^k and trainedZippedData contains w_i^{k+1}
      // Combine these two into the new train_rdd 
      val diffs: (DenseVector[Double], Double) =
        train_rdd.reduceByKey((x, y) => (y._1 - x._1, y._2 - x._2)).map(x => x._2).reduce((x, y) => (x._1 + y._1, x._2 + y._2))

      /**
       * Communication step. Communicate the new model to all nodes. Resume training.
       */
      // model.updateWeights(model.getWeights() + newDiffs._1)
      // model.updateEll(model.getEll() + newDiffs._2)
      model.updateWeights(model.getWeights() + diffs._1)
      model.updateEll(model.getEll() + diffs._2)

      val trainError = SolverUtils.averageLoss(train_data, lossFn, predictFn, model)
      val testError = SolverUtils.averageLoss(testDataUnord, lossFn, predictFn, model)

      train_rdd = trainedZippedData

      logger.info("[DATA] %d,%f,%f\n".format(roundNum, trainError, testError))
      println("[Round #%d] Test loss = %f, Train loss = %f\n".format(roundNum, testError, trainError))

    }
  }

  /**
   * ****************************************************************
   *    ___        ___   _____ ____ _      __
   *   / _ \ ____ / _ ) / ___// __/| | /| / /
   *  / // //___// _  |/ /__ / _/  | |/ |/ /
   * /____/     /____/ \___//_/    |__/|__/
   *
   * (The actual CoCoA) (try)
   * ****************************************************************
   */
  def chainDBCFCoCoAv2(): Unit = { //TODO: delete

    val NUM_ROUNDS: Int = 5
    val NUM_PART: Int = 1
    val PERC_TRAIN: Double = 0.1 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val trainDataUnord: Vector[LabeledObject] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val testDataUnord: Vector[LabeledObject] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    val conf = new SparkConf().setAppName("Chain-DBCFW").setMaster("local").set("spark.cores.max", "1")
    val sc = new SparkContext(conf)

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    // val train_data = trainDataUnord(List.fromArray(perm))
    val train_data: Array[LabeledObject] = trainDataUnord(List.fromArray(perm).slice(0, (PERC_TRAIN * trainDataUnord.size).toInt)).toArray

    val solverOptions: SolverOptions = new SolverOptions()
    solverOptions.numPasses = 1 // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = false
    solverOptions.xldebug = false
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = false

    val d: Int = featureFn(train_data(0).label, train_data(0).pattern).size
    // Let the initial model contain zeros for all weights
    var globalModel: StructSVMModel = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    /**
     *  Create two RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     */
    val indexedTrainData: Array[(Index, LabeledObject)] = (0 until train_data.size).toArray.zip(train_data)
    val indexedPrimals: Array[(Index, PrimalInfo)] = (0 until train_data.size).toArray.zip(
      Array.fill(train_data.size)((DenseVector.zeros[Double](d), 0.0)) // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
      )

    val indexedTrainDataRDD: RDD[(Index, LabeledObject)] = sc.parallelize(indexedTrainData, NUM_PART)
    var indexedPrimalsRDD: RDD[(Index, PrimalInfo)] = sc.parallelize(indexedPrimals, NUM_PART)
    
    println("Beginning training of %d data points in %d passes with lambda=%f".format(train_data.size, NUM_ROUNDS, solverOptions.lambda))

    logger.info("[DATA] round,time,train_error,test_error")
    val startTime = System.currentTimeMillis()

    for (roundNum <- 1 to NUM_ROUNDS) {

      val temp: RDD[(StructSVMModel, Array[(Index, PrimalInfo)])] = indexedTrainDataRDD.join(indexedPrimalsRDD).mapPartitions(x => DBCFWSolver.optimizeCoCoA(x, globalModel, featureFn, lossFn, oracleFn,
        predictFn, solverOptions, miniBatchEnabled=false), preservesPartitioning=true)

      val reducedData: (StructSVMModel, RDD[(Index, PrimalInfo)]) = DBCFWSolver.combineModelsCoCoA(temp, indexedPrimalsRDD, globalModel, d, beta=1.0)

      globalModel = reducedData._1
      indexedPrimalsRDD = reducedData._2

      val trainError = SolverUtils.averageLoss(DenseVector(train_data), lossFn, predictFn, globalModel)
      val testError = SolverUtils.averageLoss(testDataUnord, lossFn, predictFn, globalModel)

      val elapsedTime = (System.currentTimeMillis() - startTime).toDouble / 1000.0

      logger.info("[DATA] %d,%f,%f,%f\n".format(roundNum, elapsedTime, trainError, testError))
      println("[Round #%d] Train loss = %f, Test loss = %f\n".format(roundNum, trainError, testError))

    }

  }
  
  /**
   * ****************************************************************
   *    ___        ___   _____ ____ _      __
   *   / _ \ ____ / _ ) / ___// __/| | /| / /
   *  / // //___// _  |/ /__ / _/  | |/ |/ /
   * /____/     /____/ \___//_/    |__/|__/
   *
   * (Supports both CoCoA and miniBatch)
   * ****************************************************************
   */
  def chainDBCFCoCoACombined(): Unit = { //TODO: move to optimization package

    val NUM_ROUNDS: Int = 5
    val NUM_PART: Int = 1
    val PERC_TRAIN: Double = 0.1 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val trainDataUnord: Vector[LabeledObject] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val testDataUnord: Vector[LabeledObject] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    val conf = new SparkConf().setAppName("Chain-DBCFW").setMaster("local").set("spark.cores.max", "1")
    val sc = new SparkContext(conf)

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    // val train_data = trainDataUnord(List.fromArray(perm))
    val train_data: Array[LabeledObject] = trainDataUnord(List.fromArray(perm).slice(0, (PERC_TRAIN * trainDataUnord.size).toInt)).toArray

    val solverOptions: SolverOptions = new SolverOptions()
    solverOptions.numPasses = 1 // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = false
    solverOptions.xldebug = false
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = false

    val d: Int = featureFn(train_data(0).label, train_data(0).pattern).size
    // Let the initial model contain zeros for all weights
    var globalModel: StructSVMModel = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    /**
     *  Create two RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     */
    val indexedTrainData: Array[(Index, LabeledObject)] = (0 until train_data.size).toArray.zip(train_data)
    val indexedPrimals: Array[(Index, PrimalInfo)] = (0 until train_data.size).toArray.zip(
      Array.fill(train_data.size)((DenseVector.zeros[Double](d), 0.0)) // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
      )

    val indexedTrainDataRDD: RDD[(Index, LabeledObject)] = sc.parallelize(indexedTrainData, NUM_PART)
    var indexedPrimalsRDD: RDD[(Index, PrimalInfo)] = sc.parallelize(indexedPrimals, NUM_PART)
    
    println("Beginning training of %d data points in %d passes with lambda=%f".format(train_data.size, NUM_ROUNDS, solverOptions.lambda))

    logger.info("[DATA] round,time,train_error,test_error")
    val startTime = System.currentTimeMillis()

    for (roundNum <- 1 to NUM_ROUNDS) {

      val temp: RDD[(StructSVMModel, Array[(Index, PrimalInfo)])] = indexedTrainDataRDD.join(indexedPrimalsRDD).mapPartitions(x => DBCFWSolver.optimizeCoCoA(x, globalModel, featureFn, lossFn, oracleFn,
        predictFn, solverOptions, miniBatchEnabled=true), preservesPartitioning=true)

      val reducedData: (StructSVMModel, RDD[(Index, PrimalInfo)]) = DBCFWSolver.combineModelsCoCoA(temp, indexedPrimalsRDD, globalModel, d, beta=1.0)

      globalModel = reducedData._1
      // indexedPrimalsRDD = indexedPrimalsRDD.join(temp.flatMap(_._2)).reduce(f)
      
      val trainError = SolverUtils.averageLoss(DenseVector(train_data), lossFn, predictFn, globalModel)
      val testError = SolverUtils.averageLoss(testDataUnord, lossFn, predictFn, globalModel)

      val elapsedTime = (System.currentTimeMillis() - startTime).toDouble / 1000.0

      logger.info("[DATA] %d,%f,%f,%f\n".format(roundNum, elapsedTime, trainError, testError))
      println("[Round #%d] Train loss = %f, Test loss = %f\n".format(roundNum, trainError, testError))

    }

  }
  
  /**
   * ****************************************************************
   *    ___        ___   _____ ____ _      __
   *   / _ \ ____ / _ ) / ___// __/| | /| / /
   *  / // //___// _  |/ /__ / _/  | |/ |/ /
   * /____/     /____/ \___//_/    |__/|__/
   *
   * (Mini batch)
   * ****************************************************************
   */
  def chainDBCFCMiniBatchBuggy(): Unit = { //TODO: delete

    val NUM_ROUNDS: Int = 5
    val NUM_PART: Int = 1
    val PERC_TRAIN: Double = 0.1 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val trainDataUnord: Vector[LabeledObject] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val testDataUnord: Vector[LabeledObject] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    val conf = new SparkConf().setAppName("Chain-DBCFW").setMaster("local").set("spark.cores.max", "1")
    val sc = new SparkContext(conf)

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    // val train_data = trainDataUnord(List.fromArray(perm))
    val train_data: Array[LabeledObject] = trainDataUnord(List.fromArray(perm).slice(0, (PERC_TRAIN * trainDataUnord.size).toInt)).toArray

    val solverOptions: SolverOptions = new SolverOptions()
    solverOptions.numPasses = 1 // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = false
    solverOptions.xldebug = false
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = false

    val d: Int = featureFn(train_data(0).label, train_data(0).pattern).size
    // Let the initial model contain zeros for all weights
    var globalModel: StructSVMModel = new StructSVMModel(DenseVector.zeros(d), 0.0, DenseVector.zeros(d), featureFn, lossFn, oracleFn, predictFn)

    /**
     *  Create two RDDs:
     *  1. indexedTrainData = (Index, LabeledObject) and
     *  2. indexedPrimals (Index, Primal) where Primal = (w_i, l_i) <- This changes in each round
     */
    val indexedTrainData: Array[(Index, LabeledObject)] = (0 until train_data.size).toArray.zip(train_data)
    val indexedPrimals: Array[(Index, PrimalInfo)] = (0 until train_data.size).toArray.zip(
      Array.fill(train_data.size)((DenseVector.zeros[Double](d), 0.0)) // Fill up a list of (ZeroVector, 0.0) - the initial w_i and l_i
      )

    val indexedTrainDataRDD: RDD[(Index, LabeledObject)] = sc.parallelize(indexedTrainData, NUM_PART)
    var indexedPrimalsRDD: RDD[(Index, PrimalInfo)] = sc.parallelize(indexedPrimals, NUM_PART)
    
    println("Beginning training of %d data points in %d passes with lambda=%f".format(train_data.size, NUM_ROUNDS, solverOptions.lambda))

    logger.info("[DATA] round,time,train_error,test_error")
    val startTime = System.currentTimeMillis()

    for (roundNum <- 1 to NUM_ROUNDS) {

      val temp: RDD[(StructSVMModel, Array[(Index, PrimalInfo)])] = indexedTrainDataRDD.join(indexedPrimalsRDD).mapPartitions(x => DBCFWSolver.optimizeCoCoA(x, globalModel, featureFn, lossFn, oracleFn,
        predictFn, solverOptions, miniBatchEnabled=true), preservesPartitioning=true)
        
      // Obtained delta_w and delta_ell in previous step. Add them to previous values.
      val newPrimals: RDD[(Index, PrimalInfo)] = temp.flatMap(x => x._2).join(indexedPrimalsRDD).map(x => (x._1, (x._2._1._1 + x._2._2._1, x._2._1._2 + x._2._2._2)))

      // Obtain the new global model
      val reducedData: (StructSVMModel, RDD[(Index, PrimalInfo)]) = DBCFWSolver.combineModelsCoCoA(temp, indexedPrimalsRDD, globalModel, d, 1.0)

      globalModel = reducedData._1
      indexedPrimalsRDD = newPrimals

      val trainError = SolverUtils.averageLoss(DenseVector(train_data), lossFn, predictFn, globalModel)
      val testError = SolverUtils.averageLoss(testDataUnord, lossFn, predictFn, globalModel)

      val elapsedTime = (System.currentTimeMillis() - startTime).toDouble / 1000.0

      logger.info("[DATA] %d,%f,%f,%f\n".format(roundNum, elapsedTime, trainError, testError))
      println("[Round #%d] Train loss = %f, Test loss = %f\n".format(roundNum, trainError, testError))

    }

  }

  def main(args: Array[String]): Unit = {
    // chainDBCFW()
    // chainBCFW()
    PropertyConfigurator.configure("conf/log4j.properties")
    // chainDBCFWCoCoA()
    // chainDBCFWwAvg()
    // chainDBCFCoCoAv2()
    // chainDBCFCMiniBatch()
    chainDBCFCoCoACombined()
  }

}