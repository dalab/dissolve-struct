package ch.ethz.dal.dbcfw.classification

import breeze.linalg.DenseVector

object Types {

  type Index = Int
  type PrimalInfo = Tuple2[DenseVector[Double], Double]
  
}