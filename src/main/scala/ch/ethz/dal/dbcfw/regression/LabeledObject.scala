package ch.ethz.dal.dbcfw.regression

import org.apache.spark.mllib.regression.LabeledPoint

import breeze.linalg._

class LabeledObject(
    val label:Vector[Double],
    val pattern: Matrix[Double]) /*extends LabeledPoint*/ {
}