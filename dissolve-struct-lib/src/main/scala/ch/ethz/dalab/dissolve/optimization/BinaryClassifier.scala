package ch.ethz.dalab.dissolve.optimization

import scala.collection.mutable.HashMap
import scala.reflect.ClassTag

import org.apache.spark.mllib
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import breeze.linalg._
import ch.ethz.dalab.dissolve.models.BinarySVM
import ch.ethz.dalab.dissolve.regression.LabeledObject

class BinaryClassifier(invFreqLoss: Boolean = true)
    extends SSVMClassifier[Vector[Double], Int](new BinarySVM(HashMap(1 -> 1.0, -1 -> 1.0))) {

  override def getModel() = super.getModel()

  def convertLabPointToLabObj(dataLP: RDD[LabeledPoint],
                              isDense: Boolean): RDD[LabeledObject[Vector[Double], Int]] =
    dataLP.map {
      case lp: LabeledPoint =>
        val featureVec: Vector[Double] =
          if (isDense)
            DenseVector(lp.features.toArray)
          else {
            val sparseFeatureVec = lp.features.toSparse
            val builder: VectorBuilder[Double] = new VectorBuilder(sparseFeatureVec.indices,
              sparseFeatureVec.values,
              sparseFeatureVec.indices.length,
              lp.features.size)
            builder.toSparseVector
          }

        val label = lp.label.toInt

        LabeledObject(label, featureVec)
    }

  def convertLabPointToLabObj(dataLP: Seq[LabeledPoint],
                              isDense: Boolean): Seq[LabeledObject[Vector[Double], Int]] =
    dataLP.map {
      case lp: LabeledPoint =>
        val featureVec: Vector[Double] =
          if (isDense)
            DenseVector(lp.features.toArray)
          else {
            val sparseFeatureVec = lp.features.toSparse
            val builder: VectorBuilder[Double] = new VectorBuilder(sparseFeatureVec.indices,
              sparseFeatureVec.values,
              sparseFeatureVec.indices.length,
              lp.features.size)
            builder.toSparseVector
          }

        val label = lp.label.toInt

        LabeledObject(label, featureVec)
    }

  /**
   * Distributed Optimization
   */
  override def train(trainData: RDD[LabeledObject[Vector[Double], Int]],
                     testData: Option[RDD[LabeledObject[Vector[Double], Int]]],
                     solver: DistributedSolver[Vector[Double], Int])(implicit m: ClassTag[Int]): Unit = {
    weights = solver.train(trainData, testData)
  }

  def train(trainDataLP: RDD[LabeledPoint],
            testDataLP: RDD[LabeledPoint],
            solver: DistributedSolver[Vector[Double], Int]): Unit =
    train(trainDataLP, Some(testDataLP), solver)

  def train(trainDataLP: RDD[LabeledPoint],
            testDataLP: Option[RDD[LabeledPoint]],
            solver: DistributedSolver[Vector[Double], Int]): Unit = {

    // Check if vectors are dense or sparse
    val x = trainDataLP.take(1)(0)
    val isDenseFeatures = x.features match {
      case _: mllib.linalg.DenseVector  => true
      case _: mllib.linalg.SparseVector => false
      case _                            => throw new Exception("")
    }

    // Verify sanity of labels
    // In this case, we want the labels to be [0, numClasses)
    val classCount = trainDataLP.map(_.label.toInt)
      .groupBy(identity)
      .mapValues(_.size)
      .collect()
      .toMap
    val thisLabels: List[Int] = classCount.keys.toList.sorted
    val validLabels: List[Int] = List(-1, 1)
    val isValidLabels = thisLabels.zip(validLabels).forall { case (t, v) => t == v }
    assert(isValidLabels, "Expected labels = %s, Found labels = %s".format(validLabels, thisLabels))

    // Convert type LabeledPoint to LabeledObject
    val trainDataRDD = convertLabPointToLabObj(trainDataLP, isDenseFeatures)
    val testDataRDD = testDataLP match {
      case Some(testData) => Some(convertLabPointToLabObj(testData, isDenseFeatures))
      case None           => None
    }

    // Additionally, calculate class frequencies from trainData
    if (invFreqLoss) {

      val totalCount = classCount.values.sum.toDouble
      val classFreq = classCount.mapValues(_ / totalCount)

      model match {
        case bsvmModel: BinarySVM =>
          bsvmModel.setClassFreq(-1, classFreq(-1))
          bsvmModel.setClassFreq(+1, classFreq(+1))
        case _ => throw new Exception("Unrecognized Binary SVM Model")
      }
    }

    train(trainDataRDD, testDataRDD, solver)
  }

  /**
   * Local Optimization
   */
  override def train(trainData: Seq[LabeledObject[Vector[Double], Int]],
                     testData: Option[Seq[LabeledObject[Vector[Double], Int]]],
                     solver: LocalSolver[Vector[Double], Int])(implicit m: ClassTag[Int]): Unit = {

    // calculate class frequencies from trainData
    if (invFreqLoss) {
      val classCount = trainData.map(_.label)
        .groupBy(identity)
        .mapValues(_.size)
        .toMap

      val thisLabels: List[Int] = classCount.keys.toList.sorted
      val validLabels: List[Int] = List(-1, 1)

      val isValidLabels = thisLabels.zip(validLabels).forall { case (t, v) => t == v }
      assert(isValidLabels, "Expected labels = %s, Found labels = %s".format(validLabels, thisLabels))

      val totalCount = classCount.values.sum.toDouble
      val classFreq = classCount.mapValues(_ / totalCount)

      model match {
        case bsvmModel: BinarySVM =>
          bsvmModel.setClassFreq(-1, classFreq(-1))
          bsvmModel.setClassFreq(+1, classFreq(+1))
        case _ => throw new Exception("Unrecognized Binary SVM Model")
      }
    }

    weights = solver.train(trainData, testData)
  }

}