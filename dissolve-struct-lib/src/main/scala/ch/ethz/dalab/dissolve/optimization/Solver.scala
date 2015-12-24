package ch.ethz.dalab.dissolve.optimization

import scala.reflect.ClassTag
import org.apache.spark.rdd.RDD
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.classification.MutableWeightsEll
import breeze.linalg._

trait Solver[X, Y] extends Serializable {

  def train(trainData: RDD[LabeledObject[X, Y]],
            testData: Option[RDD[LabeledObject[X, Y]]])(implicit m: ClassTag[Y]): Vector[Double]

  def train(trainData: Seq[LabeledObject[X, Y]],
            testData: Option[Seq[LabeledObject[X, Y]]])(implicit m: ClassTag[Y]): Vector[Double]

}