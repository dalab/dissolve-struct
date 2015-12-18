package ch.ethz.dalab.dissolve.optimization

import org.apache.spark.rdd.RDD
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import scala.reflect.ClassTag
import breeze.linalg.Vector

trait DistributedSolver[X, Y] extends Solver[X, Y] {

  def train(data: RDD[LabeledObject[X, Y]])(implicit m: ClassTag[Y]): Vector[Double] =
    train(data, Option.empty)

  def train(trainData: RDD[LabeledObject[X, Y]],
            testData: RDD[LabeledObject[X, Y]])(implicit m: ClassTag[Y]): Vector[Double] = train(trainData, Some(testData))

  def train(trainData: Seq[LabeledObject[X, Y]],
            testData: Option[Seq[LabeledObject[X, Y]]])(implicit m: ClassTag[Y]): Vector[Double] =
    throw new UnsupportedOperationException("DistributedSolver does not support Local Data")

}