package ch.ethz.dalab.dissolve.models

import ch.ethz.dalab.dissolve.classification.Types._
import ch.ethz.dalab.dissolve.optimization.DissolveFunctions

import breeze.linalg._

import cc.factorie.infer.MaximizeByBPLoopy
import cc.factorie.la.DenseTensor1
import cc.factorie.la.Tensor
import cc.factorie.model._
import cc.factorie.singleFactorIterable
import cc.factorie.variable.DiscreteDomain
import cc.factorie.variable.DiscreteVariable

import scala.collection.mutable.ArrayBuffer

/**
 * Each object represents an object X composed of nodes { x_i }
 * x_i is the data node, containing local features
 * i \in [0, nNodes)
 */
class GNodes(
  // Graph structure
  // Vertices numbered : [0 - nNodes)
  // Edges
  val adjacencyList: Array[Array[Index]],
  // x_i's
  val localFeatures: Array[Vector[Double]]) extends Serializable

/**
 * Represents labels for nodes
 * Each object represents an object Y composed of nodes { y_i }
 * y_i is the label for the i'th node, y_i \in [0, nStates)
 * i \in [0, nNodes)
 */
class GLabels(
  // Vertices numbered : [0 - nNodes)
  // Labels for each y_i
  val labelList: Array[Label]) extends Serializable

class GraphCRF(nNodes: Int,
               nStates: Int,
               disablePairwise: Boolean = false,
               invFreqLoss: Map[Int, Double] = null) extends DissolveFunctions[GNodes, GLabels] {

  /**
   * Feature map \phi
   */
  def featureFn(xM: GNodes, yM: GLabels): Vector[Double] = {

    val d = xM.localFeatures(0).size
    val labels = yM.labelList

    val unaryFeatures: DenseVector[Double] = DenseVector.zeros(nStates * d)
    for (idx <- 0 until nNodes) {
      val label = labels(idx)

      val localStartIdx = label * d
      val localEndIdx = localStartIdx + d

      val x_local = xM.localFeatures(idx)

      unaryFeatures(localStartIdx until localEndIdx) :+= x_local
    }

    val pairwiseFeatures: DenseVector[Double] =
      if (disablePairwise)
        DenseVector.zeros(0)
      else {
        val transitions = DenseMatrix.zeros[Double](nStates, nStates)

        // Iterate over each node i
        for (thisN <- 0 until nNodes) {
          val thisLabel = labels(thisN)

          // Iterate over each node j adjacent to i
          xM.adjacencyList(thisN).foreach {
            case thatN =>
              val thatLabel = labels(thatN)

              transitions(thisLabel, thatLabel) += 1.0
              transitions(thatLabel, thisLabel) += 1.0
          }
        }

        normalize(transitions.toDenseVector, 2)
      }

    val featureVec = DenseVector.vertcat(unaryFeatures, pairwiseFeatures)

    featureVec
  }

  /**
   * Per label loss \delta
   */
  def perLabelLoss(labTruth: Label, labPredict: Label): Double =
    if (labTruth == labPredict)
      0.0
    else if (invFreqLoss != null) {
      1.0 / invFreqLoss(labTruth)
    } else
      1.0

  /**
   * Structured loss \Delta
   */
  def lossFn(yTruth: GLabels, yPredict: GLabels): Double = {

    assert(yTruth.labelList.size == yPredict.labelList.size,
      "Failed: yTruth.labelList.size == yPredict.labelList.size")

    val trueLabels = yTruth.labelList
    val predLabels = yPredict.labelList

    val indvLosses =
      trueLabels.zip(predLabels)
        .map { case (labTruth, labPred) => perLabelLoss(labTruth, labPred) }

    indvLosses.sum / indvLosses.size
  }

  /**
   * Construct Factor graph and run MAP
   * (Max-product using Loopy Belief Propogation)
   */
  def decode(unaryPot: DenseMatrix[Double],
             pairwisePot: DenseMatrix[Double],
             adj: Array[Array[Index]]): Array[Label] = {

    val _nNodes = unaryPot.cols
    val _nStates = unaryPot.rows

    assert(nNodes == _nNodes, "nNodes mismatch: %d == %d.".format(nNodes, _nNodes))
    assert(nStates == _nStates, "nStates mismatch: %d == %d.".format(nStates, _nStates))

    if (!disablePairwise)
      assert(pairwisePot.rows == nStates)

    object PixelDomain extends DiscreteDomain(nStates)

    class Pixel(i: Int) extends DiscreteVariable(i) {
      def domain = PixelDomain
    }

    def getUnaryFactor(yi: Pixel, idx: Int): Factor = {
      new Factor1(yi) {
        val weights: DenseTensor1 = new DenseTensor1(unaryPot(::, idx).toArray)
        def score(k: Pixel#Value) = unaryPot(k.intValue, idx)
        override def valuesScore(tensor: Tensor): Double = {
          weights dot tensor
        }
      }
    }

    def getPairwiseFactor(yi: Pixel, yj: Pixel): Factor = {
      new Factor2(yi, yj) {
        val weights: DenseTensor1 = new DenseTensor1(pairwisePot.toArray)
        def score(i: Pixel#Value, j: Pixel#Value) = pairwisePot(i.intValue, j.intValue)
        override def valuesScore(tensor: Tensor): Double = {
          weights dot tensor
        }
      }
    }

    val pixelSeq: IndexedSeq[Pixel] =
      (0 until nNodes).map(x => new Pixel(12))

    val unaryFactors: IndexedSeq[Factor] =
      (0 until nNodes).map {
        case idx =>
          getUnaryFactor(pixelSeq(idx), idx)
      }

    val model = new ItemizedModel
    model ++= unaryFactors

    if (!disablePairwise) {
      val pairwiseFactors =
        (0 until nNodes).flatMap {
          case thisIdx =>
            val thisFactors = new ArrayBuffer[Factor]

            adj(thisIdx).foreach {
              case nextIdx =>
                thisFactors ++=
                  getPairwiseFactor(pixelSeq(thisIdx), pixelSeq(nextIdx))
            }
            thisFactors
        }
      model ++= pairwiseFactors
    }

    MaximizeByBPLoopy.maximize(pixelSeq, model)

    val mapLabels: Array[Label] = (0 until nNodes).map {
      idx =>
        pixelSeq(idx).intValue
    }.toArray

    mapLabels
  }

  def unpackWeightVec(weightv: DenseVector[Double], d: Int): (DenseMatrix[Double], DenseMatrix[Double]) = {

    assert(weightv.size >= (nStates * d))

    val unaryWeights = weightv(0 until nStates * d)
    val unaryWeightMat = unaryWeights.toDenseMatrix.reshape(d, nStates)

    val pairwisePot =
      if (!disablePairwise) {
        assert(weightv.size == (nStates * d) + (nStates * nStates))
        val pairwiseWeights = weightv((nStates * d) until weightv.size)
        pairwiseWeights.toDenseMatrix.reshape(nStates, nStates)
      } else null

    (unaryWeightMat, pairwisePot)
  }

  /**
   * Maximization Oracle
   */
  override def oracleFn(weightVec: Vector[Double], xi: GNodes, yi: GLabels): GLabels = {

    val d = xi.localFeatures.size
    val (unaryPot, pairwisePot) = unpackWeightVec(weightVec.toDenseVector, d)

    // Loss-augment the scores
    if (yi != null) {
      for (superIdx <- 0 until nNodes) {
        val trueLabel = yi.labelList(superIdx)

        for (predLabel <- 0 until nStates)
          unaryPot(predLabel, superIdx) += perLabelLoss(trueLabel, predLabel) / nNodes
      }
    }

    // Decode
    val t0 = System.currentTimeMillis()
    val decodedLabels = decode(unaryPot, pairwisePot, xi.adjacencyList)
    val oracleSol = new GLabels(decodedLabels)
    val t1 = System.currentTimeMillis()

    oracleSol
  }

  /**
   * Prediction function
   */
  def predictFn(weightVec: Vector[Double], xi: GNodes): GLabels = {
    oracleFn(weightVec, xi, null)
  }

}