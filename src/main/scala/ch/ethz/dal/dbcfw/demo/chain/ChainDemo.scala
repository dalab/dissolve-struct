package ch.ethz.dal.dbcfw.demo.chain

import ch.ethz.dal.dbcfw.regression.LabeledObject
import breeze.linalg._
import breeze.numerics._
import breeze.generic._
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import ch.ethz.dal.dbcfw.classification.StructSVMWithSSG

object ChainDemo {

  val debugOn = true

  /**
   * Reads data produced by the convert-ocr-data.py script and loads into memory as a vector of Labeled objects
   *
   *  TODO
   *  * Take foldNumber as a parameter and return training and test set
   */
  def loadData(patternsFilename: String, labelsFilename: String, foldFilename: String): Vector[LabeledObject] = {
    val patterns: Array[String] = scala.io.Source.fromFile(patternsFilename).getLines().toArray[String]
    val labels: Array[String] = scala.io.Source.fromFile(labelsFilename).getLines().toArray[String]
    val folds: Array[String] = scala.io.Source.fromFile(foldFilename).getLines().toArray[String]

    val n = labels.size

    assert(patterns.size == labels.size, "#Patterns=%d, but #Labels=%d".format(patterns.size, labels.size))
    assert(patterns.size == folds.size, "#Patterns=%d, but #Folds=%d".format(patterns.size, folds.size))

    val data: Vector[LabeledObject] = DenseVector.fill(n) { null }

    for (i ← 0 until n) {
      // Expected format: id, #rows, #cols, (pixels_i_j,)* pixels_n_m
      val patLine: List[Double] = patterns(i).split(",").map(x ⇒ x.toDouble) toList
      // Expected format: id, #letters, (letters_i)* letters_n
      val labLine: List[Double] = labels(i).split(",").map(x ⇒ x.toDouble) toList

      val patNumRows: Int = patLine(1) toInt
      val patNumCols: Int = patLine(2) toInt

      val patVals: Array[Double] = patLine.slice(3, patLine.size).toArray[Double]
      // The pixel values should be Column-major ordered
      val thisPattern: Matrix[Double] = DenseVector(patVals).toDenseMatrix.reshape(patNumRows, patNumCols)

      val labVals: Array[Double] = labLine.slice(2, labLine.size).toArray[Double]
      val thisLabel: DenseVector[Double] = DenseVector(labVals)

      data(i) = new LabeledObject(thisLabel, thisPattern)
    }

    data
  }

  /**
   * Returns a vector, capturing unary, bias and pairwise features of the word
   */
  def featureFn(y: Vector[Double], x: Matrix[Double]): Vector[Double] = {
    val numStates = 26
    val numDims = x.rows
    val numVars = x.cols
    val phi: Vector[Double] = DenseVector.zeros(numStates * numDims + 2 * numStates + numStates * numStates)

    /* Unaries */
    for (i ← 0 until numVars) {
      val idx = (y(i).toInt * numDims)
      // Wanted: x(::, i). But, looks like there is no clean way to do this
      val temp1 = x(0 until x.rows, i to i).toDenseMatrix.toDenseVector
      val temp2 = phi((idx + 1) to (idx + 1 + numDims - 1))
      phi((idx + 1) to (idx + 1 + numDims - 1)) := phi((idx + 1) to (idx + 1 + numDims - 1)) + temp1
    }

    phi(numStates * numDims + y(0).toInt + 1) = 1.0
    phi(numStates * numDims + numStates + y(-1).toInt + 1) = 1.0

    /* Pairwise */
    val offset = numStates * numDims + 2 * numStates
    for (i ← 0 until numVars - 1) {
      val idx = y(i).toInt + numStates * y(i + 1).toInt
      phi(offset + idx) = phi(offset + idx) + 1
    }

    phi
  }

  /**
   * Return Normalized Hamming distance
   */
  def lossFn(yTruth: Vector[Double], yPredict: Vector[Double]): Double =
    sum((yTruth :== yPredict).map(x ⇒ if (x) 1 else 0)) / yTruth.size.toDouble

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
  def columnwiseMax(mat: Matrix[Double]): DenseMatrix[Double] = {
    val colMax: DenseMatrix[Double] = DenseMatrix.zeros[Double](2, mat.cols)

    for (col ← 0 until mat.cols) {
      // 1st row contains max
      colMax(0, col) = max(colMax(::, col))
      // 2nd row contains indices of the max
      colMax(1, col) = argmax(colMax(::, col))
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
    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](nNodes, nStates)
    val mxState: DenseMatrix[Double] = DenseMatrix.zeros[Double](nNodes, nStates)
    alpha(0, ::) := logNodePot(0, ::)
    for (n ← 1 until nNodes) {
      /* Equivalent to `tmp = repmat(alpha(n-1, :)', 1, nStates) + logEdgePot` */
      // Create an empty 26x26 repmat term
      val alphaRepmat: DenseMatrix[Double] = DenseMatrix.zeros[Double](nStates, nStates)
      for (row ← 0 until nStates) {
        // Take the (n-1)th row from alpha and represent it as a column in repMat
        // alpha(n-1, ::) returns a Transposed view, so use the below workaround
        alphaRepmat(::, row) := alpha.t(::, n - 1)
      }
      val tmp: DenseMatrix[Double] = alphaRepmat + logEdgePot
      val colMaxTmp: DenseMatrix[Double] = columnwiseMax(tmp)
      alpha(n, ::) := logNodePot(n, ::) + colMaxTmp(0, ::)
      mxState(n, ::) := colMaxTmp(1, ::)
    }

    /*--- Backward pass ---*/
    val y: DenseVector[Double] = DenseVector.zeros[Double](nNodes)
    // [dummy, y(nNodes)] = max(alpha(nNodes, :))
    y(nNodes - 1) = argmax(alpha.t(::, nNodes - 1))
    for (n ← nNodes - 2 to 0 by -1) {
      y(n) = mxState(n + 1, y(n + 1).toInt)
    }
    y
  }

  /**
   * The Maximization Oracle
   */
  def oracleFn(model: StructSVMModel, yi: Vector[Double], xiM: Matrix[Double]): Vector[Double] = {
    val numStates = 26
    val xi = xiM.toDenseMatrix
    val numDims = xi.rows
    val numVars = xi.cols

    val weight: Weight = weightVecToObj(model.getWeights(), numStates, numDims)

    val thetaUnary: DenseMatrix[Double] = weight.unary.t * xi // Produces a 129x9 matrix

    // First position has a bias
    thetaUnary(::, 0) := thetaUnary(::, 0) + weight.firstBias
    // Last position has a bias
    thetaUnary(::, -1) := thetaUnary(::, -1) + weight.lastBias

    val thetaPairwise: DenseMatrix[Double] = weight.pairwise

    // Add loss-augmentation to the score (normalized Hamming distances used for loss)
    val l: Int = yi.size
    for (i ← 0 until numVars) {
      thetaUnary(::, i) := thetaUnary(::, i) + 1.0 / l
      val idx = yi(i).toInt
      thetaUnary(idx, i) = thetaUnary(idx, i) - 1.0 / l
    }

    // Solve inference problem
    val label: Vector[Double] = logDecode(thetaUnary.t, thetaPairwise) - 1.0

    label
  }

  def predictFn(model: StructSVMModel, xiM: Matrix[Double]): Vector[Double] = {
    val numStates = 26
    val xi = xiM.toDenseMatrix
    val numDims = xi.rows
    val numVars = xi.cols

    val weight: Weight = weightVecToObj(model.getWeights(), numStates, numDims)

    val thetaUnary: DenseMatrix[Double] = weight.unary.t * xi // Produces a 129x9 matrix

    // First position has a bias
    thetaUnary(::, 0) := thetaUnary(::, 0) + weight.firstBias
    // Last position has a bias
    thetaUnary(::, -1) := thetaUnary(::, -1) + weight.lastBias

    val thetaPairwise: DenseMatrix[Double] = weight.pairwise

    // Solve inference problem
    val label: Vector[Double] = logDecode(thetaUnary.t, thetaPairwise) - 1.0

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

  def main(args: Array[String]): Unit = {

    val data: Vector[LabeledObject] = loadData("data/ocr-patterns.csv", "data/ocr-labels.csv", "data/ocr-folds.csv")

    if (debugOn)
      println("Loaded %d examples, pattern:%dx%d and labels:%dx1"
        .format(data.size,
          data(0).pattern.rows,
          data(1).pattern.cols,
          data(0).label.size))

    // Fix seed for reproducibility
    util.Random.setSeed(1)

    // Split data into training and test datasets
    val trnPrc = 0.80
    val perm: List[Int] = util.Random.shuffle((0 until data.size) toList)
    val cutoffIndex: Int = (trnPrc * perm.size) toInt
    val train_data = data(perm.slice(0, cutoffIndex)) toVector // Obtain in range [0, cutoffIndex)
    val test_data = data(perm.slice(cutoffIndex, perm.size)) toVector // Obtain in range [cutoffIndex, data.size)

    val trainer: StructSVMWithSSG = new StructSVMWithSSG(train_data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn)
      .withNumPasses(1)
      .withRegularizer(0.01)

    val model: StructSVMModel = trainer.trainModel()

    var truePredictions = 0
    val totalPredictions = test_data.size

    for (item ← test_data) {
      val prediction = model.predictFn(model, item.pattern)
      println("True = %-10s\tPrediction = %-10s".format(labelVectorToString(item.label), labelVectorToString(prediction)))
      // println("True = %d\tPrediction=%d".format(item.label.size, prediction.size))
      if (prediction == item.label)
        truePredictions += 1
    }

  }

}