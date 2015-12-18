package ch.ethz.dalab.dissolve.optimization

import scala.reflect.ClassTag
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import org.apache.spark.rdd.RDD
import breeze.linalg.Vector

trait LocalSolver[X, Y] extends Solver[X, Y] {

  def train(data: Seq[LabeledObject[X, Y]])(implicit m: ClassTag[Y]): Vector[Double] =
    train(data, Option.empty)

  def train(trainData: Seq[LabeledObject[X, Y]],
            testData: Seq[LabeledObject[X, Y]])(implicit m: ClassTag[Y]): Vector[Double] =
    train(trainData, Some(testData))

  def train(trainData: RDD[LabeledObject[X, Y]],
            testData: Option[RDD[LabeledObject[X, Y]]])(implicit m: ClassTag[Y]): Vector[Double] =
    throw new UnsupportedOperationException("LocalSolver does not support Distributed Data (RDDs)")

}