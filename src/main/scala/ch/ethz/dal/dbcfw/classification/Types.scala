package ch.ethz.dal.dbcfw.classification

import breeze.linalg.DenseVector

object Types {

  type Index = Int
  type Primal = Tuple2[DenseVector[Double], Double]
  
}