package ch.ethz.dalab.dissolve.models

import breeze.linalg._
import cc.factorie.infer.MaximizeByBPChain
import cc.factorie.la.DenseTensor1
import cc.factorie.la.Tensor
import cc.factorie.model.Factor
import cc.factorie.model.Factor1
import cc.factorie.model.Factor2
import cc.factorie.model.ItemizedModel
import cc.factorie.variable.DiscreteDomain
import cc.factorie.variable.DiscreteVariable
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions

/**
 * A Linear Chain CRF model with observed data `x` and hidden labels `y`
 *
 * @param disablePairwise Disable pairwise interactions between labels. Default = false.
 * @param useBPDecoding Use Belief Propagation from Factorie for decoding. Default = Viterbi decoding.
 */
class LinearChainCRF(numStates: Int,
                     disablePairwise: Boolean = false,
                     useBPDecoding: Boolean = false,
                     normalizedHammingLoss: Boolean = true) extends DissolveFunctions[Matrix[Double], Vector[Double]] {

  val ENABLE_PERF_METRICS = false

  def time[R](block: => R, blockName: String = ""): R = {
    if (ENABLE_PERF_METRICS) {
      val t0 = System.currentTimeMillis()
      val result = block // call-by-name
      val t1 = System.currentTimeMillis()
      println("%25s %d ms".format(blockName, (t1 - t0)))
      result
    } else block
  }

  /**
   * Returns a vector, capturing unary, bias and pairwise features of the word
   *
   * x is a Pattern matrix, of dimensions numDims x NumVars (say 129 x 9)
   * y is a Label vector, of dimension numVars (9 for above x)
   */
  def featureFn(xM: Matrix[Double], y: Vector[Double]): Vector[Double] = {
    val x = xM.toDenseMatrix
    val numDims = x.rows // 129 in case of Chain OCR
    val numVars = x.cols
    // First term for unaries, Second term for first and last letter biases, Third term for Pairwise features
    // Unaries are row-major ordered, i.e., [0,129) positions for 'a', [129, 258) for 'b' and so on 
    val phi: DenseVector[Double] =
      if (!disablePairwise)
        DenseVector.zeros[Double]((numStates * numDims) + (2 * numStates) + (numStates * numStates))
      else
        DenseVector.zeros[Double]((numStates * numDims) + (2 * numStates))

    /* Unaries */
    for (i <- 0 until numVars) {
      val idx = (y(i).toInt * numDims)
      phi((idx) until (idx + numDims)) :=
        phi((idx) until (idx + numDims)) + x(::, i)
    }

    phi(numStates * numDims + y(0).toInt) = 1.0
    phi(numStates * numDims + numStates + y(-1).toInt) = 1.0

    /* Pairwise */
    if (!disablePairwise) {
      val offset = (numStates * numDims) + (2 * numStates)
      for (i <- 0 until (numVars - 1)) {
        val idx = y(i).toInt + numStates * y(i + 1).toInt
        phi(offset + idx) = phi(offset + idx) + 1.0
      }
    }

    phi
  }

  /**
   * Return Normalized Hamming distance
   */
  def lossFn(yTruth: Vector[Double], yPredict: Vector[Double]): Double = {
    val sumIndivLoss =
      sum((yTruth :== yPredict).map(x => if (x) 0 else 1)).toDouble

    if (normalizedHammingLoss)
      sumIndivLoss / yTruth.size
    else
      sumIndivLoss
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

  def unaryOnlyDecode(logNodePotMat: Matrix[Double]): Vector[Double] = {
    val maxUnaryPotChainMat = columnwiseMax(logNodePotMat)
    // First row contains potentials
    // Second row contains indices
    val decodedChain = maxUnaryPotChainMat(1, ::).t

    decodedChain
  }

  /**
   * Viterbi decoding, with forward and backward passes
   * (works for both loss-augmented or not, just takes the given potentials)
   */
  def viterbiDecode(logNodePotMat: Matrix[Double], logEdgePotMat: Matrix[Double]): Vector[Double] = {

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
   * Chain Belief Propagation Decoding
   * Alternate decoding function using belief propagation on the factor graph,
   * using the Factorie library.
   */
  def bpDecode(thetaUnary: Matrix[Double], thetaPairwise: Matrix[Double]): Vector[Double] = {
    // thetaUnary is a (lengthOfChain x 26) dimensional matrix
    val nNodes: Int = thetaUnary.rows
    val nStates: Int = thetaUnary.cols

    val label: DenseVector[Double] = DenseVector.zeros[Double](nNodes)

    val unaryPot = thetaUnary.toDenseMatrix.t
    val pairwisePot = thetaPairwise.toDenseMatrix

    object LetterDomain extends DiscreteDomain(nStates)

    class LetterVar(i: Int) extends DiscreteVariable(i) {
      def domain = LetterDomain
    }

    def getUnaryFactor(yi: LetterVar, posInChain: Int): Factor = {
      new Factor1(yi) {
        val weights: DenseTensor1 = new DenseTensor1(unaryPot(::, posInChain).toArray)
        def score(i: LetterVar#Value) = unaryPot(posInChain, i.intValue)
        override def valuesScore(tensor: Tensor): Double = {
          weights dot tensor
        }
      }
    }

    def getPairwiseFactor(yi: LetterVar, yj: LetterVar): Factor = {
      new Factor2(yi, yj) {
        val weights: DenseTensor1 = new DenseTensor1(pairwisePot.toArray)
        def score(i: LetterVar#Value, j: LetterVar#Value) = pairwisePot(i.intValue, j.intValue)
        override def valuesScore(tensor: Tensor): Double = {
          weights dot tensor
        }
      }
    }

    val letterChain: IndexedSeq[LetterVar] = for (i <- 0 until nNodes) yield new LetterVar(0)

    val unaries: IndexedSeq[Factor] = for (i <- 0 until nNodes) yield getUnaryFactor(letterChain(i), i)
    val pairwise: IndexedSeq[Factor] = for (i <- 0 until nNodes - 1) yield getPairwiseFactor(letterChain(i), letterChain(i + 1))

    val model = new ItemizedModel
    model ++= unaries
    model ++= pairwise

    val bpMapSummary = MaximizeByBPChain(letterChain, model)
    for (i <- 0 until nNodes)
      label(i) = letterChain(i).intValue.toDouble

    label
  }

  /**
   * The Maximization Oracle
   *
   * Performs loss-augmented decoding on a given example xi and label yi
   * using model.getWeights() as parameter. The loss is normalized Hamming loss.
   *
   * If yi is not given, then standard prediction is done (i.e. MAP decoding),
   * without any loss term.
   */
  def oracleFnWithDecode(model: StructSVMModel[Matrix[Double], Vector[Double]], xi: Matrix[Double], yi: Vector[Double],
                         decodeFn: (Matrix[Double], Matrix[Double]) => Vector[Double]): Vector[Double] = {
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
    if (yi != null) { // loss augmentation is only done if a label yi is given. 
      val l: Int =
        if (normalizedHammingLoss) yi.size else 1
      for (i <- 0 until numVars) {
        thetaUnary(::, i) := thetaUnary(::, i) + 1.0 / l
        val idx = yi(i).toInt // Loss augmentation
        thetaUnary(idx, i) = thetaUnary(idx, i) - 1.0 / l
      }
    }

    // Solve inference problem
    val label: Vector[Double] =
      time({
        if (!disablePairwise)
          decodeFn(thetaUnary.t, thetaPairwise) // - 1.0
        else
          unaryOnlyDecode(thetaUnary)
      }, "Decode ")

    label
  }

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
    val pairwise: DenseMatrix[Double] =
      if (!disablePairwise)
        weightVec((idx + 2 * numStates) until weightVec.size)
          .toDenseVector
          .toDenseMatrix
          .reshape(numStates, numStates)
      else
        null

    new Weight(unary, firstBias.toDenseVector, lastBias.toDenseVector, pairwise)
  }

  override def oracleFn(model: StructSVMModel[Matrix[Double], Vector[Double]], xi: Matrix[Double], yi: Vector[Double]): Vector[Double] =
    if (useBPDecoding)
      oracleFnWithDecode(model, xi, yi, bpDecode)
    else
      oracleFnWithDecode(model, xi, yi, viterbiDecode)

  /**
   * Predict function.
   * This is (non-loss-augmented) decoding
   *
   */
  def predictFn(model: StructSVMModel[Matrix[Double], Vector[Double]], xi: Matrix[Double]): Vector[Double] = {
    val label: Vector[Double] = oracleFn(model, xi, null)

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

}