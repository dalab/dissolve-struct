package ch.ethz.dalab.dissolve.optimization

import scala.reflect.ClassTag
import ch.ethz.dalab.dissolve.regression.LabeledObject
import ch.ethz.dalab.dissolve.classification.StructSVMModel

trait LocalSolver[X, Y] {

  def train(data: Seq[LabeledObject[X, Y]])(implicit m: ClassTag[Y]): StructSVMModel[X, Y] =
    train(data, Option.empty)

  def train(trainData: Seq[LabeledObject[X, Y]],
            testData: Seq[LabeledObject[X, Y]])(implicit m: ClassTag[Y]): StructSVMModel[X, Y] =
    train(trainData, Some(testData))

  /**
   * When testData is specified, test/validation error is computed
   * per debug round as specified by `debugMultiplier` option
   */
  def train(trainData: Seq[LabeledObject[X, Y]],
            testData: Option[Seq[LabeledObject[X, Y]]])(implicit m: ClassTag[Y]): StructSVMModel[X, Y]

}