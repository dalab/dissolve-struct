package ch.ethz.dal.dissolve.examples.imageseg

import org.apache.log4j.PropertyConfigurator
import breeze.linalg.{ Matrix, Vector }
import ch.ethz.dal.dbcfw.classification.StructSVMModel
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix

case class ROIFeature(feature: Vector[Double]) // Represent each pixel/region by a feature vector

case class ROILabel(label: Int, numClasses: Int = 24) {
  override def equals(o: Any) = o match {
    case that: ROILabel => that.label == this.label
    case _              => false
  }
}

object ImageSegmentationDemo {

  /**
   * Feature Function.
   * Uses: http://arxiv.org/pdf/1408.6804v2.pdf
   */
  def featureFn(yMat: Matrix[ROILabel], xMat: Matrix[ROIFeature]): Vector[Double] = {

    assert(xMat.rows == yMat.rows)
    assert(xMat.cols == yMat.cols)

    val x = xMat.toDenseMatrix.toDenseVector
    val y = yMat.toDenseMatrix.toDenseVector

    val numFeatures = x(0).feature.size
    val numClasses = y(0).numClasses
    val numRegions = x.size

    val unary = DenseMatrix.zeros[Double](numFeatures * numRegions, numClasses)
    val pairwise = DenseMatrix.zeros[Double](numClasses, numClasses)

    // Populate the unary features
    for (classNum <- 0 until numClasses) {

      // For each class label, zero-out the x_i whose class label does not agree
      val xTimesIndicator = x.toArray
        .zipWithIndex
        .flatMap {
          case (roiFeature, idx) =>
            if (classNum == y(idx)) // Compare this feature's label
              roiFeature.feature.toArray
            else
              Array.fill(numFeatures)(0.0)
        }

      val startIdx = classNum * numFeatures * numRegions
      val endIdx = (classNum + 1) * numFeatures * numRegions

      unary(::, classNum) := DenseVector(xTimesIndicator)

    }

    // Populate the pairwise features
    for (
      i <- 1 until xMat.cols - 1;
      j <- 1 until xMat.rows - 1
    ) {
      val classA = yMat(i, j).label

      for (
        delx <- List(-1, 0, 1);
        dely <- List(-1, 0, 1) if ((delx != 0) && (dely != 0))
      ) {
        val classB = yMat(i + delx, j + dely).label

        pairwise(classA, classB) += 1.0
        pairwise(classB, classA) += 1.0
      }
    }

    DenseVector.vertcat(unary.toDenseVector, pairwise.toDenseVector)
  }

  /**
   * Loss function
   */
  def lossFn(yTruth: Matrix[ROILabel], yPredict: Matrix[ROILabel]): Double = {

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
  def oracleFn(model: StructSVMModel[Matrix[ROIFeature], Matrix[ROILabel]], yi: Matrix[ROILabel], xi: Matrix[ROIFeature]): Matrix[ROILabel] = {

    assert(xi.rows == yi.rows)
    assert(xi.cols == yi.cols)

    val numClasses = yi(0, 0).numClasses
    val numRows = xi.rows
    val numCols = xi.cols
    val numROI = numRows * numCols
    val xFeatureSize = xi(0, 0).feature.size

    val weightVec = model.getWeights()

    val unaryStartIdx = 0
    val unaryEndIdx = xFeatureSize * numROI * numClasses
    val unary: DenseMatrix[Double] = weightVec(unaryStartIdx until unaryEndIdx)
      .toDenseVector
      .toDenseMatrix
      .reshape(xFeatureSize * numROI, numClasses)

    val pairwiseStartIdx = unaryEndIdx
    val pairwiseEndIdx = weightVec.size
    assert(pairwiseEndIdx - pairwiseStartIdx == numClasses * numClasses)
    val pairwise: DenseMatrix[Double] = weightVec(pairwiseStartIdx until pairwiseEndIdx)
      .toDenseVector
      .toDenseMatrix
      .reshape(numClasses, numClasses)

    // object ROIDomain extends DiscreteDomain(numClasses)

    null
  }

  /**
   * Prediction Function
   */
  def predictFn(model: StructSVMModel[Matrix[ROIFeature], Matrix[ROILabel]], xi: Matrix[ROIFeature]): Matrix[ROILabel] = {

    null
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