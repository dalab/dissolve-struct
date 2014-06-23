package ch.ethz.dal.dbcfw.demo.chain

import ch.ethz.dal.dbcfw.regression.LabeledObject

import breeze.linalg._
import breeze.numerics._

object ChainDemo {

  val debugOn = true

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

  def main(args: Array[String]): Unit = {

    val data: Vector[LabeledObject] = loadData("data/ocr-patterns.csv", "data/ocr-labels.csv", "data/ocr-folds.csv")

    if (debugOn)
      println("Loaded %d examples, pattern:%dx%d and labels:%dx1"
        .format(data.size,
          data(0).pattern.rows,
          data(1).pattern.cols,
          data(0).label.size))

  }

}