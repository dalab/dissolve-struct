package ch.ethz.dal.dbcfw.demo.chain

import breeze.linalg._
import breeze.numerics._
import breeze.generic._

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD

import org.apache.log4j.Logger
import org.apache.log4j.PropertyConfigurator

import java.io.File
import java.io.PrintWriter

import collection.JavaConversions.enumerationAsScalaIterator

import ch.ethz.dal.dbcfw.regression.LabeledObject
import ch.ethz.dal.dbcfw.classification.Types._
import ch.ethz.dal.dbcfw.classification.StructSVMWithDBCFW
import ch.ethz.dal.dbcfw.optimization.DBCFWSolver
import ch.ethz.dal.dbcfw.classification.StructSVMWithMiniBatch
import ch.ethz.dal.dbcfw.optimization.SolverUtils
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import ch.ethz.dal.dbcfw.classification.StructSVMWithSSG
import ch.ethz.dal.dbcfw.optimization.SolverOptions
import ch.ethz.dal.dbcfw.classification.StructSVMWithBCFW

import cc.factorie._
import cc.factorie.variable._
import cc.factorie.model._
import cc.factorie.infer.MaximizeByBP
import cc.factorie.infer.MaximizeByBPChain
import cc.factorie.infer.BP
import cc.factorie.infer.MaximizeByBPLoopy
import cc.factorie.infer.MaximizeByMPLP

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
  def loadData(patternsFilename: String, labelsFilename: String, foldFilename: String): DenseVector[LabeledObject[Matrix[Double], Vector[Double]]] = {
    val patterns: Array[String] = scala.io.Source.fromFile(patternsFilename).getLines().toArray[String]
    val labels: Array[String] = scala.io.Source.fromFile(labelsFilename).getLines().toArray[String]
    val folds: Array[String] = scala.io.Source.fromFile(foldFilename).getLines().toArray[String]

    val n = labels.size

    assert(patterns.size == labels.size, "#Patterns=%d, but #Labels=%d".format(patterns.size, labels.size))
    assert(patterns.size == folds.size, "#Patterns=%d, but #Folds=%d".format(patterns.size, folds.size))

    val data: DenseVector[LabeledObject[Matrix[Double], Vector[Double]]] = DenseVector.fill(n) { null }

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
   * (works for both loss-augmented or not, just takes the given potentials)
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
   * Alternate decoding function using Factor Graphs
   */
  def bpDecode(thetaUnary: Matrix[Double], thetaPairwise: Matrix[Double]): Vector[Double] = {
    // thetaUnary is a (lengthOfChain x 26) dimensional matrix
    val nNodes: Int = thetaUnary.rows
    val nStates: Int = thetaUnary.cols

    val label: DenseVector[Double] = DenseVector.zeros[Double](nNodes)

    object LetterDomain extends DiscreteDomain(nStates)

    class LetterVar(i: Int) extends DiscreteVariable(i) {
      def domain = LetterDomain
    }

    def getUnaryFactor(yi: LetterVar, posInChain: Int): Factor = {
      new Factor1(yi) {
        def score(i: LetterVar#Value) = thetaUnary(posInChain, i.intValue)
      }
    }

    def getPairwiseFactor(yi: LetterVar, yj: LetterVar): Factor = {
      new Factor2(yi, yj) {
        def score(i: LetterVar#Value, j: LetterVar#Value) = thetaPairwise(i.intValue, j.intValue)
      }
    }

    val letterChain: IndexedSeq[LetterVar] = for (i <- 0 until nNodes) yield new LetterVar(0)

    val unaries: IndexedSeq[Factor] = for (i <- 0 until nNodes) yield getUnaryFactor(letterChain(i), i)
    val pairwise: IndexedSeq[Factor] = for (i <- 0 until nNodes - 1) yield getPairwiseFactor(letterChain(i), letterChain(i + 1))

    val model = new ItemizedModel
    model ++= unaries
    model ++= pairwise

    // val m = BP.inferChainMax(letterChain, model)
    val assgn = MaximizeByMPLP.infer(letterChain, model).mapAssignment
    for (i <- 0 until nNodes)
      label(i) = assgn(letterChain(i)).intValue.toDouble

    label
  }

  /**
   * The Maximization Oracle
   * y_i (true label) is used for Loss Augmentation
   */
  def oracleFnWithDecode(model: StructSVMModel[Matrix[Double], Vector[Double]], yi: Vector[Double], xi: Matrix[Double],
    decodeFn: (Matrix[Double], Matrix[Double]) => Vector[Double]): Vector[Double] = {
    val numStates = 26
    // val xi = xiM.toDenseMatrix // 129 x n matrix, ex. 129 x 9 if len(word) = 9
    val numDims = xi.rows // 129 in Chain example 
    val numVars = xi.cols // The length of word, say 9

    // Convert the lengthy weight vector into an object, to ease representation
    // weight.unary is a numDims x numStates Matrix (129 x 26 in above example)
    // weight.firstBias and weight.lastBias is a numStates-dimensional vector
    // weight.pairwise is a numStates x numStates Matrix
    val weight: Weight = weightVecToObj(model.getWeights(), numStates, numDims)

    val thetaUnary: DenseMatrix[Double] = weight.unary.t * xi // Produces a 26 x (length-of-chain) matrix

    // First position has a bias
    thetaUnary(::, 0) := thetaUnary(::, 0) + weight.firstBias
    // Last position has a bias
    thetaUnary(::, -1) := thetaUnary(::, -1) + weight.lastBias

    val thetaPairwise: DenseMatrix[Double] = weight.pairwise

    // Add loss-augmentation to the score (normalized Hamming distances used for loss)
    val l: Int = yi.size
    for (i <- 0 until numVars) {
      thetaUnary(::, i) := thetaUnary(::, i) + 1.0 / l
      val idx = yi(i).toInt // Loss augmentation
      thetaUnary(idx, i) = thetaUnary(idx, i) - 1.0 / l
    }

    // Solve inference problem
    val label: Vector[Double] = decodeFn(thetaUnary.t, thetaPairwise) // - 1.0

    label
  }

  def oracleFn(model: StructSVMModel[Matrix[Double], Vector[Double]], yi: Vector[Double], xi: Matrix[Double]): Vector[Double] =
    oracleFnWithDecode(model, yi, xi, logDecode)
  
  def oracleFnBP(model: StructSVMModel[Matrix[Double], Vector[Double]], yi: Vector[Double], xi: Matrix[Double]): Vector[Double] =
    oracleFnWithDecode(model, yi, xi, bpDecode)
    
  /**
   * Predict (this could use (non-loss-augmented) decoding
   *
   * TODO
   * * Use MaxOracle instead of this (Use yi: Option<Vector[Double]>)
   */
  def predictFnWithDecode(model: StructSVMModel[Matrix[Double], Vector[Double]], xi: Matrix[Double],
    decodeFn: (Matrix[Double], Matrix[Double]) => Vector[Double]): Vector[Double] = {
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
    val label: Vector[Double] = decodeFn(thetaUnary.t, thetaPairwise) // - 1.0

    label
  }

  def predictFn(model: StructSVMModel[Matrix[Double], Vector[Double]], xi: Matrix[Double]): Vector[Double] = {
    predictFnWithDecode(model, xi, logDecode)
  }

  def predictFnBP(model: StructSVMModel[Matrix[Double], Vector[Double]], xi: Matrix[Double]): Vector[Double] = {
    predictFnWithDecode(model, xi, bpDecode)
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
   * ****************************************************************
   */
  def chainBCFW(): Unit = {

    val PERC_TRAIN: Double = 0.05 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val train_data_unord: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val test_data: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    // val train_data = train_data_unord(List.fromArray(perm))
    val train_data: Array[LabeledObject[Matrix[Double], Vector[Double]]] = train_data_unord(List.fromArray(perm).slice(0, (PERC_TRAIN * train_data_unord.size).toInt)).toArray
    // val temp: DenseVector[LabeledObject] = train_data_unord(List.fromArray(perm).slice(0, 1)).toDenseVector
    // val train_data = DenseVector.fill(5){temp(0)}

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

    val solverOptions: SolverOptions[Matrix[Double], Vector[Double]] = new SolverOptions();
    solverOptions.numPasses = 3
    solverOptions.debug = true
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = true
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = true
    solverOptions.testData = Some(test_data.toArray)
    
    solverOptions.enableOracleCache = false
    solverOptions.oracleCacheSize = 10
    
    solverOptions.debugInfoPath = "/Users/tribhu/git/DBCFWstruct/debug/debug-bcfw-%d.csv".format(System.currentTimeMillis())
    
    /*val trainer: StructSVMWithSSG = new StructSVMWithSSG(train_data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)*/

    val trainer: StructSVMWithBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithBCFW[Matrix[Double], Vector[Double]](train_data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)

    val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

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
   * (CoCoA)
   * ****************************************************************
   */
  def chainDBCFWCoCoA(options: Map[String, String]): Unit = {

    /**
     * Load all options
     */
    val PERC_TRAIN: Double = options.getOrElse("perctrain", "0.05").toDouble // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)
    
    val appName: String = options.getOrElse("appname", "Chain-DBCFW")

    val solverOptions: SolverOptions[Matrix[Double], Vector[Double]] = new SolverOptions()
    solverOptions.numPasses = options.getOrElse("numpasses", "5").toInt // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean
    solverOptions.lambda = options.getOrElse("lambda", "0.01").toDouble
    solverOptions.doWeightedAveraging = options.getOrElse("wavg", "false").toBoolean
    solverOptions.doLineSearch = options.getOrElse("linesearch", "true").toBoolean
    solverOptions.debugLoss = options.getOrElse("debugloss", "false").toBoolean

    solverOptions.sample = options.getOrElse("sample", "frac")
    solverOptions.sampleFrac = options.getOrElse("samplefrac", "0.5").toDouble
    solverOptions.sampleWithReplacement = options.getOrElse("samplewithreplacement", "false").toBoolean
    
    solverOptions.enableManualPartitionSize = options.getOrElse("manualrddpart", "false").toBoolean
    solverOptions.NUM_PART = options.getOrElse("numpart", "2").toInt
    solverOptions.autoconfigure = options.getOrElse("autoconfigure", "false").toBoolean
    
    solverOptions.enableOracleCache = options.getOrElse("enableoracle", "false").toBoolean
    solverOptions.oracleCacheSize = options.getOrElse("oraclesize", "5").toInt
    
    /**
     * Some local overrides
     */
    solverOptions.sampleFrac = 1.0
    solverOptions.enableOracleCache = false
    solverOptions.oracleCacheSize = 10
    solverOptions.numPasses = 3
    solverOptions.enableManualPartitionSize = true
    solverOptions.NUM_PART = 1
    solverOptions.doWeightedAveraging = true
    
    solverOptions.debugInfoPath = "/Users/tribhu/git/DBCFWstruct/debug/debug-dissolve-%d.csv".format(System.currentTimeMillis())
    
    println("# PERC_TRAIN=%f".format(PERC_TRAIN))
    println(solverOptions.toString())

    /**
     * Begin execution
     */
    val trainDataUnord: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val testDataUnord: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    println("Loaded data with %d rows, pattern=%dx%d, label=%dx1".format(trainDataUnord.size, trainDataUnord(0).pattern.rows, trainDataUnord(0).pattern.cols, trainDataUnord(0).label.size))

    val conf = new SparkConf().setAppName(appName).setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")
    
    println(SolverUtils.getSparkConfString(sc.getConf))

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    val train_data: Array[LabeledObject[Matrix[Double], Vector[Double]]] = trainDataUnord(List.fromArray(perm).slice(0, (PERC_TRAIN * trainDataUnord.size).toInt)).toArray
    
    solverOptions.testDataRDD = Some(sc.parallelize(testDataUnord.toArray, solverOptions.NUM_PART))

    val trainer: StructSVMWithDBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithDBCFW[Matrix[Double], Vector[Double]](
      sc.parallelize(train_data, solverOptions.NUM_PART),
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)

    val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

    var avgTrainLoss: Double = 0.0
    for (item <- train_data) {
      val prediction = model.predictFn(model, item.pattern)
      avgTrainLoss += lossFn(item.label, prediction)
    }
    println("Average loss on training set = %f".format(avgTrainLoss / train_data.size))

    var avgTestLoss: Double = 0.0
    for (item <- testDataUnord) {
      val prediction = model.predictFn(model, item.pattern)
      avgTestLoss += lossFn(item.label, prediction)
    }
    println("Average loss on test set = %f".format(avgTestLoss / testDataUnord.size))

  }

  /*
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
  def chainDBCFWMiniBatch(): Unit = {

    val PERC_TRAIN: Double = 0.1 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val trainDataUnord: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val testDataUnord: Vector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    println("Loaded data with %d rows, pattern=%dx%d, label=%dx1".format(trainDataUnord.size, trainDataUnord(0).pattern.rows, trainDataUnord(0).pattern.cols, trainDataUnord(0).label.size))

    val conf = new SparkConf().setAppName("Chain-DBCFW")
    val sc = new SparkContext(conf)

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    // val train_data = trainDataUnord(List.fromArray(perm))
    val train_data: Array[LabeledObject[Matrix[Double], Vector[Double]]] = trainDataUnord(List.fromArray(perm).slice(0, (PERC_TRAIN * trainDataUnord.size).toInt)).toArray

    val solverOptions: SolverOptions[Matrix[Double], Vector[Double]] = new SolverOptions()
    solverOptions.numPasses = 20 // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = false
    solverOptions.xldebug = false
    solverOptions.lambda = 0.01
    solverOptions.doWeightedAveraging = false
    solverOptions.doLineSearch = true
    solverOptions.debugLoss = false
    solverOptions.testData = testDataUnord

    solverOptions.H = train_data.size
    solverOptions.NUM_PART = 2

    val trainer: StructSVMWithMiniBatch = new StructSVMWithMiniBatch(sc,
      DenseVector(train_data),
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)

    val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

    var avgTrainLoss: Double = 0.0
    for (item <- train_data) {
      val prediction = model.predictFn(model, item.pattern)
      avgTrainLoss += lossFn(item.label, prediction)
    }
    println("Average loss on training set = %f".format(avgTrainLoss / train_data.size))

    var avgTestLoss: Double = 0.0
    for (item <- testDataUnord) {
      val prediction = model.predictFn(model, item.pattern)
      avgTestLoss += lossFn(item.label, prediction)
    }
    println("Average loss on test set = %f".format(avgTestLoss / testDataUnord.size))

  }
  */

  
  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("conf/log4j.properties")

    val options: Map[String, String] = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt) => (opt -> "true")
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap
    
    System.setProperty("spark.akka.frameSize","512");
    println(options)

    chainDBCFWCoCoA(options)

    chainBCFW()
  }

}