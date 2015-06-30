package ch.ethz.dalab.dissolve.examples.imageseg

import breeze.linalg._
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions

/**
 * `data` is a `F` x `N` matrix; each column contains the features for a super-pixel
 * `transitions` is an adjacency list. transitions(i) contains neighbours of super-pixel `i+1`'th super-pixel
 * `pixelMapping` contains mapping of super-pixel to corresponding pixel in the original image (column-major)
 */
case class QuantizedImage(unaries: DenseMatrix[Double],
                          pairwise: Array[Array[Int]],
                          pixelMapping: Array[Int],
                          width: Int,
                          height: Int,
                          filename: String = "NA")
/**
 * labels(i) contains label for `i+1`th super-pixel
 */
case class QuantizedLabel(labels: Array[Int],
                          filename: String = "NA")

/**
 * Functions for dissolve^struct
 */
object ImageSegmentationAdv extends DissolveFunctions[QuantizedImage, QuantizedLabel] {

  def featureFn(xMat: QuantizedImage, yMat: QuantizedLabel): Vector[Double] = ???

  def lossFn(yTruth: QuantizedLabel, yPredict: QuantizedLabel): Double = ???

  override def oracleFn(model: StructSVMModel[QuantizedImage, QuantizedLabel], xi: QuantizedImage, yi: QuantizedLabel): QuantizedLabel = ???

  def predictFn(model: StructSVMModel[QuantizedImage, QuantizedLabel], xi: QuantizedImage): QuantizedLabel = {
    oracleFn(model, xi, null)
  }

}