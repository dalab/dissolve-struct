package ch.ethz.dalab.dissolve.examples.imageseg

import org.apache.log4j.PropertyConfigurator
import breeze.linalg.{ Matrix, Vector }
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix
import cc.factorie.variable.DiscreteDomain
import cc.factorie.variable.DiscreteVariable
import breeze.linalg.sum
import breeze.linalg.Axis
import cc.factorie.model.Factor1
import cc.factorie.model.Factor
import cc.factorie.model.Factor2
import cc.factorie.model.ItemizedModel
import cc.factorie.infer.MaximizeByMPLP

case class ROIFeature(feature: Vector[Double]) // Represent each pixel/region by a feature vector

case class ROILabel(label: Int, numClasses: Int = 24) {
  override def equals(o: Any) = o match {
    case that: ROILabel => that.label == this.label
    case _              => false
  }
}

object ImageSegmentationDemo {

  /**
   * Given:
   * - a matrix xMat of super-pixels, of size r = n x m, and x_i, an f-dimensional vector
   * - corresponding labels of these super-pixels yMat, with K classes
   * Return:
   * - a matrix of size (f*K) x r, i.e, each column corresponds to the feature map of x_i
   */
  def getUnaryFeatureMap(yMat: DenseMatrix[ROILabel], xMat: DenseMatrix[ROIFeature]): DenseMatrix[Double] = {
    assert(xMat.rows == yMat.rows)
    assert(xMat.cols == yMat.cols)

    println("xMat.size = %d x %d".format(xMat.rows, xMat.cols))

    val numFeatures = xMat(0, 0).feature.size // f
    val numClasses = yMat(0, 0).numClasses // K
    val numRegions = xMat.rows * xMat.cols // r

    val unaryMat = DenseMatrix.zeros[Double](numFeatures * numClasses, numRegions)

    /**
     * Populate unary features
     * For each node i in graph defined by xMat, whose feature vector is x_i and corresponding label is y_i,
     * construct a feature map phi_i given by: [I(y_i = 0)x_i I(y_i = 1)x_i ... I(y_i = K)x_i ]
     */
    
    xMat.keysIterator.foreach {
      case (r, c) =>
        val i = r * xMat.rows + c // Column-major iteration

        // println("(r, c) = (%d, %d)".format(r, c))
        val x_i = xMat(r, c).feature
        val y_i = yMat(r, c).label

        val phi_i = DenseVector.zeros[Double](numFeatures * numClasses)

        val startIdx = numFeatures * y_i
        val endIdx = startIdx + numFeatures

        // For y_i'th position of phi_i, populate x_i's feature vector
        phi_i(startIdx until endIdx) := x_i

        unaryMat(::, i) := phi_i
    }

    unaryMat
  }

  /**
   * Given:
   * - a matrix xMat of super-pixels, of size r = n x m, and x_i, an f-dimensional vector
   * - corresponding labels of these super-pixels yMat, with K classes
   * return:
   * - a matrix of size K x K, capturing pairwise scores
   */
  def getPairwiseFeatureMap(yMat: DenseMatrix[ROILabel], xMat: DenseMatrix[ROIFeature]): DenseMatrix[Double] = {
    assert(xMat.rows == yMat.rows)
    assert(xMat.cols == yMat.cols)

    val numFeatures = xMat(0, 0).feature.size
    val numClasses = yMat(0, 0).numClasses
    val numRegions = xMat.rows * xMat.cols

    val pairwiseMat = DenseMatrix.zeros[Double](numClasses, numClasses)

    for (
      c <- 1 until xMat.cols - 1;
      r <- 1 until xMat.rows - 1
    ) {
      val classA = yMat(r, c).label

      // Iterate all neighbours
      for (
        delx <- List(-1, 0, 1);
        dely <- List(-1, 0, 1) if ((delx != 0) && (dely != 0))
      ) {
        val classB = yMat(r + delx, c + dely).label

        pairwiseMat(classA, classB) += 1.0
        pairwiseMat(classB, classA) += 1.0
      }
    }

    pairwiseMat
  }

  /**
   * Feature Function.
   * Uses: http://arxiv.org/pdf/1408.6804v2.pdf
   * http://www.kev-smith.com/papers/LUCCHI_ECCV12.pdf
   */
  def featureFn(yMat: DenseMatrix[ROILabel], xMat: DenseMatrix[ROIFeature]): Vector[Double] = {

    assert(xMat.rows == yMat.rows)
    assert(xMat.cols == yMat.cols)

    val unaryMat = getUnaryFeatureMap(yMat, xMat)
    val pairwiseMat = getPairwiseFeatureMap(yMat, xMat)

    // Collapse feature function of each x_i by addition
    val unarySumVec = sum(unaryMat, Axis._1)

    // Double-check dimensions
    val numClasses = yMat(0, 0).numClasses
    val numFeatures = xMat(0, 0).feature.size
    assert(unarySumVec.size == numClasses * numFeatures)
    assert(pairwiseMat.size == numClasses * numClasses)

    DenseVector.vertcat(unarySumVec, pairwiseMat.toDenseVector)
  }

  /**
   * Loss function
   */
  def lossFn(yTruth: DenseMatrix[ROILabel], yPredict: DenseMatrix[ROILabel]): Double = {

    assert(yTruth.rows == yPredict.rows)
    assert(yTruth.cols == yPredict.cols)

    val loss =
      for (
        x <- 0 until yTruth.cols;
        y <- 0 until yTruth.rows
      ) yield {
        if (x == y) 0.0 else 1.0
      }

    loss.sum
  }

  /**
   * Oracle function
   */
  def oracleFn(model: StructSVMModel[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]], yi: DenseMatrix[ROILabel], xi: DenseMatrix[ROIFeature]): DenseMatrix[ROILabel] = {

    assert(xi.rows == yi.rows)
    assert(xi.cols == yi.cols)

    val numClasses = yi(0, 0).numClasses
    val numRows = xi.rows
    val numCols = xi.cols
    val numROI = numRows * numCols
    val xFeatureSize = xi(0, 0).feature.size

    val weightVec = model.getWeights()

    val unaryStartIdx = 0
    val unaryEndIdx = xFeatureSize * numClasses
    // This should be a vector of dimensions 1 x (K x h), where K = #Classes, h = dim(x_i)
    val unary: DenseVector[Double] = weightVec(unaryStartIdx until unaryEndIdx).toDenseVector
    val unaryMat: DenseMatrix[Double] = unary.toDenseMatrix.reshape(numClasses, xFeatureSize).t // Returns a K x h matrix, i.e, class to feature mapping

    val pairwiseStartIdx = unaryEndIdx
    val pairwiseEndIdx = weightVec.size
    assert(pairwiseEndIdx - pairwiseStartIdx == numClasses * numClasses)
    val pairwise: DenseMatrix[Double] = weightVec(pairwiseStartIdx until pairwiseEndIdx)
      .toDenseVector
      .toDenseMatrix
      .reshape(numClasses, numClasses)

    val phi_Y: DenseMatrix[Double] = getUnaryFeatureMap(yi, xi) // Retrieves a (K x h) x m matrix
    val thetaUnary = phi_Y * unary // Construct a (1 x m) vector
    val thetaPairwise = pairwise

    /**
     * Parameter estimation
     */
    object ROIDomain extends DiscreteDomain(numClasses)

    class ROIClassVar(i: Int) extends DiscreteVariable(i) {
      def domain = ROIDomain
    }

    def getUnaryFactor(yi: ROIClassVar, x: Int, y: Int): Factor = {
      new Factor1(yi) {
        def score(i: ROIClassVar#Value) = {
          val w_yi = unaryMat(::, yi.intValue)
          val unaryPotAtxy = xi(x, y).feature dot w_yi

          unaryPotAtxy
        }
      }
    }

    def getPairwiseFactor(yi: ROIClassVar, yj: ROIClassVar): Factor = {
      new Factor2(yi, yj) {
        def score(i: ROIClassVar#Value, j: ROIClassVar#Value) = thetaPairwise(i.intValue, j.intValue)
      }
    }

    val letterChain: IndexedSeq[ROIClassVar] = for (i <- 0 until numROI) yield new ROIClassVar(0)

    val unaryFactors: IndexedSeq[Factor] = for (i <- 0 until numROI) yield {
      // Column-major strides
      val colNum = i / numCols
      val rowNum = i % numRows
      getUnaryFactor(letterChain(i), rowNum, colNum)
    }

    val pairwiseFactors: IndexedSeq[Factor] = for (i <- 0 until numROI - 1) yield getPairwiseFactor(letterChain(i), letterChain(i + 1))

    val bpmodel = new ItemizedModel
    bpmodel ++= unaryFactors
    bpmodel ++= pairwiseFactors

    val label: DenseVector[Int] = DenseVector.zeros[Int](numROI)
    val assgn = MaximizeByMPLP.infer(letterChain, bpmodel).mapAssignment
    for (i <- 0 until numROI)
      label(i) = assgn(letterChain(i)).intValue

    // Convert these inferred labels into an image-class mask
    val imgMask = DenseMatrix.zeros[ROILabel](numRows, numCols)
    for (
      r <- 0 until numRows;
      c <- 0 until numCols
    ) {
      val idx = r * (numRows - 1) + c
      imgMask(r, c) = ROILabel(label(idx))
    }

    imgMask
  }

  /**
   * Prediction Function
   */
  def predictFn(model: StructSVMModel[Matrix[ROIFeature], Matrix[ROILabel]], xi: Matrix[ROIFeature]): Matrix[ROILabel] = {

    val numClasses = 24
    val numRows = xi.rows
    val numCols = xi.cols
    val numROI = numRows * numCols
    val xFeatureSize = xi(0, 0).feature.size

    val weightVec = model.getWeights()

    val unaryStartIdx = 0
    val unaryEndIdx = xFeatureSize * numClasses
    // This should be a vector of dimensions 1 x (K x h), where K = #Classes, h = dim(x_i)
    val unary: DenseVector[Double] = weightVec(unaryStartIdx until unaryEndIdx).toDenseVector
    val unaryMat: DenseMatrix[Double] = unary.toDenseMatrix.reshape(numClasses, xFeatureSize).t // Returns a K x h matrix, i.e, class to feature mapping

    val pairwiseStartIdx = unaryEndIdx
    val pairwiseEndIdx = weightVec.size
    assert(pairwiseEndIdx - pairwiseStartIdx == numClasses * numClasses)
    val pairwise: DenseMatrix[Double] = weightVec(pairwiseStartIdx until pairwiseEndIdx)
      .toDenseVector
      .toDenseMatrix
      .reshape(numClasses, numClasses)

    val thetaPairwise = pairwise

    /**
     * Parameter estimation
     */
    object ROIDomain extends DiscreteDomain(numClasses)

    class ROIClassVar(i: Int) extends DiscreteVariable(i) {
      def domain = ROIDomain
    }

    def getUnaryFactor(yi: ROIClassVar, x: Int, y: Int): Factor = {
      new Factor1(yi) {
        def score(i: ROIClassVar#Value) = {
          val w_yi = unaryMat(::, yi.intValue)
          val unaryPotAtxy = xi(x, y).feature dot w_yi

          unaryPotAtxy
        }
      }
    }

    def getPairwiseFactor(yi: ROIClassVar, yj: ROIClassVar): Factor = {
      new Factor2(yi, yj) {
        def score(i: ROIClassVar#Value, j: ROIClassVar#Value) = thetaPairwise(i.intValue, j.intValue)
      }
    }

    val letterChain: IndexedSeq[ROIClassVar] = for (i <- 0 until numROI) yield new ROIClassVar(0)

    val unaryFactors: IndexedSeq[Factor] = for (i <- 0 until numROI) yield {
      // Column-major strides
      val colNum = i / numCols
      val rowNum = i % numRows
      getUnaryFactor(letterChain(i), rowNum, colNum)
    }

    val pairwiseFactors: IndexedSeq[Factor] = for (i <- 0 until numROI - 1) yield getPairwiseFactor(letterChain(i), letterChain(i + 1))

    val bpmodel = new ItemizedModel
    bpmodel ++= unaryFactors
    bpmodel ++= pairwiseFactors

    val label: DenseVector[Int] = DenseVector.zeros[Int](numROI)
    val assgn = MaximizeByMPLP.infer(letterChain, bpmodel).mapAssignment
    for (i <- 0 until numROI)
      label(i) = assgn(letterChain(i)).intValue

    // Convert these inferred labels into an image-class mask
    val imgMask = DenseMatrix.zeros[ROILabel](numRows, numCols)
    for (
      r <- 0 until numRows;
      c <- 0 until numCols
    ) {
      val idx = r * (numRows - 1) + c
      imgMask(r, c) = ROILabel(label(idx))
    }

    imgMask
  }

  def dissolveImageSementation(options: Map[String, String]) {

  }

  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("conf/log4j.properties")

    val options: Map[String, String] = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt)    => (opt -> "true")
        case _             => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    System.setProperty("spark.akka.frameSize", "512")
    println(options)

  }

}