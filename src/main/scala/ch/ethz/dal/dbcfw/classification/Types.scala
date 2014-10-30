package ch.ethz.dal.dbcfw.classification

import breeze.linalg.{Vector, DenseVector}
import scala.collection.mutable.MutableList

object Types {

  type Index = Int
  type PrimalInfo = Tuple2[DenseVector[Double], Double]
  type BoundedCacheList = MutableList[Vector[Double]]
  
}