package ch.ethz.dalab.dissolve.optimization

import org.apache.spark.rdd.RDD
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import scala.reflect.ClassTag

trait DistributedSolver[X, Y] extends Serializable {

  def train(data: RDD[LabeledObject[X, Y]])(implicit m: ClassTag[Y]): StructSVMModel[X, Y] =
    train(data, Option.empty)

  def train(trainData: RDD[LabeledObject[X, Y]],
            testData: RDD[LabeledObject[X, Y]])(implicit m: ClassTag[Y]): StructSVMModel[X, Y] = train(trainData, Some(testData))

  /**
   * When testData is specified, test/validation error is computed
   * per debug round as specified by `debugMultiplier` option
   */
  def train(trainData: RDD[LabeledObject[X, Y]],
            testData: Option[RDD[LabeledObject[X, Y]]])(implicit m: ClassTag[Y]): StructSVMModel[X, Y]

}