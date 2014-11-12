package ch.ethz.dal.dbcfw.regression

import org.apache.spark.mllib.regression.LabeledPoint

import breeze.linalg._

class LabeledObject[X, Y](
    val label: Y,
    val pattern: X) /*extends LabeledPoint*/ extends Serializable {
}