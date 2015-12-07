package ch.ethz.dalab.dissolve.classification

import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.PairRDDFunctions
import ch.ethz.dalab.dissolve.regression.LabeledObject
import scala.collection.mutable.HashMap
import ch.ethz.dalab.dissolve.regression.LabeledObject
import scala.reflect.ClassTag
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.regression.LabeledObject
import scala.collection.mutable.MutableList
import ch.ethz.dalab.dissolve.regression.LabeledObject
import org.apache.spark.mllib.regression.LabeledPoint

object ClassificationUtils {

  /**
   * Generates class weights. If classWeights is flase the default value of 1.0 is used. Otherwise if the user submitted a custom weight array
   * the weights in there will be used. If the user did not submit a custom array of weights the inverse class freq. will be used
   */
  def generateClassWeights[X, Y: ClassTag](data: RDD[LabeledObject[X, Y]], classWeights: Boolean = true, customWeights: Option[HashMap[Y,Double]] = None): HashMap[Y, Double] = {
    val map = HashMap[Y, Double]()
    val labels: Array[Y] = data.map { x: LabeledObject[X, Y] => x.label }.distinct().collect()
    if (classWeights) {
      if (customWeights.getOrElse(null) == null) {
        //inverse class frequency as weight
        val classOccur: PairRDDFunctions[Y, Double] = data.map(x => (x.label, 1.0))
        val labelOccur: PairRDDFunctions[Y, Double] = classOccur.reduceByKey((x, y) => x + y)
        val labelWeight: PairRDDFunctions[Y, Double] = labelOccur.mapValues { x => 1 / x }

        val weightSum: Double = labelWeight.values.sum()
        val nClasses: Int = labels.length
        val scaleValue: Double = nClasses / weightSum

        var sum: Double = 0.0
        for ((label, weight) <- labelWeight.collectAsMap()) {
          val clWeight = scaleValue * weight
          sum += clWeight
          map.put(label, clWeight)
        }

        assert(sum == nClasses)
      } else {
        //use custom weights
        assert(labels.length == customWeights.get.size)
        for (label <-  labels) {
          map.put(label, customWeights.get(label))
        }
      }
    } else {
      // default weight of 1.0
      for (label <- labels) {
        map.put(label, 1.0)
      }
    }
    map
  }

  def resample[X,Y:ClassTag](data: RDD[LabeledObject[X, Y]],nSamples:HashMap[Y,Int],nSlices:Int): RDD[LabeledObject[X, Y]] = {
    val buckets: HashMap[Y, RDD[LabeledObject[X, Y]]] = HashMap()
    val newData = MutableList[LabeledObject[X, Y]]()

    val labels: Array[Y] = data.map { x => x.label }.distinct().collect()

    labels.foreach { x => buckets.put(x, data.filter { point => point.label == x }) }

    for (cls <- buckets.keySet) {
      val sampledData = buckets.get(cls).get.takeSample(true, nSamples.get(cls).get)
      for (x: LabeledObject[X, Y] <- sampledData) {
        newData.+=(x)
      }
    }
    data.context.parallelize(newData, nSlices)
  }
  
  def resample(data: RDD[LabeledPoint],nSamples:HashMap[Double,Int],nSlices:Int): RDD[LabeledPoint] = {
    val buckets: HashMap[Double, RDD[LabeledPoint]] = HashMap()
    val newData = MutableList[LabeledPoint]()
    
    val labels: Array[Double] = data.map { x => x.label }.distinct().collect()

    labels.foreach { x => buckets.put(x, data.filter { point => point.label == x }) }

    for (cls <- buckets.keySet) {
      val sampledData = buckets.get(cls).get.takeSample(true, nSamples.get(cls).get)
      for (x: LabeledPoint <- sampledData) {
        newData.+=(x)
      }
    }
    data.context.parallelize(newData, nSlices)
  }

}