package ch.ethz.dal.dbcfw.utils

import breeze.linalg._

object LinAlgOps {

  def updateColumn[T](mat: Matrix[T], vec: Vector[T], colnum: Int): Matrix[T] = {

    val updatedMatrix =
      mat match {
        case sparseMat: CSCMatrix[T] =>
          for ((idx, value) <- vec.activeIterator) {
            sparseMat(idx, colnum) = value
          }
          sparseMat
        case denseMat: DenseMatrix[T] =>
          denseMat(::, colnum) := vec
          denseMat
      }

    updatedMatrix
  }

  def getMatrixColumn(mat: Matrix[Double], colnum: Int): Vector[Double] = {
    val slicedVector =
      mat match {
        case sparseMat: CSCMatrix[Double] =>
          val slicedSparseVector = SparseVector.zeros[Double](mat.rows)
          // Sparse Matrix doesn't support slicing into vector.
          // So, first slice and obtain a sparse matrix. And then transform this sparse matrix into a vector
          val foo = sparseMat(0 until mat.rows, colnum to colnum)
          for (((r, c), v) <- foo.activeIterator) {
            slicedSparseVector(r) = v
          }
          slicedSparseVector
        case denseMat: DenseMatrix[Double] =>
          denseMat(::, colnum)
      }

    slicedVector
  }

  def main(args: Array[String]): Unit = {}

}