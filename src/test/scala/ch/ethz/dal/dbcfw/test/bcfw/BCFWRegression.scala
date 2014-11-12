package ch.ethz.dal.dbcfw.test.bcfw

import ch.ethz.dal.dbcfw.demo.chain.ChainDemo._
import org.scalatest.FunSpec
import ch.ethz.dal.dbcfw.regression.LabeledObject
import breeze.linalg._
import breeze.numerics._
import ch.ethz.dal.dbcfw.optimization.SolverOptions
import ch.ethz.dal.dbcfw.classification.StructSVMWithBCFW
import ch.ethz.dal.dbcfw.classification.StructSVMModel

class BCFWRegression extends FunSpec {

  describe("BCFW-SSVM-OCR") {

    val train_data_unord: DenseVector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_train.csv", "data/labels_train.csv", "data/folds_train.csv")
    val test_data: DenseVector[LabeledObject[Matrix[Double], Vector[Double]]] = loadData("data/patterns_test.csv", "data/labels_test.csv", "data/folds_test.csv")

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    val train_data = train_data_unord(List.fromArray(perm))

    /**
     * Data files for cross-checking
     */
    val debugWeightsMatlabPattern = "data/debug/debugWeights/matlab-weights-%d.csv"
    val debugWeightsScalaPattern = "data/debug/debugWeights/scala-w-%d.csv"

    /**
     * Run BCFW solver with N passes, by writing weights after each pass. Then, compare with given data.
     */
    it("should match Matlab's weights and ells. With LineSearch and WeightedAveraging over 5 passes.") {

      val rmseThreshold = 1.0

      val solverOptions: SolverOptions[Matrix[Double], Vector[Double]] = new SolverOptions();
      solverOptions.numPasses = 5
      solverOptions.lambda = 0.01
      solverOptions.doWeightedAveraging = true
      solverOptions.doLineSearch = true

      solverOptions.debug = true
      solverOptions.xldebug = false
      solverOptions.debugWeights = true

      val trainer: StructSVMWithBCFW[Matrix[Double], Vector[Double]] = new StructSVMWithBCFW(train_data,
        featureFn,
        lossFn,
        oracleFn,
        predictFn,
        solverOptions)

      val model: StructSVMModel[Matrix[Double], Vector[Double]] = trainer.trainModel()

      // Verify if weights are similar in all passes
      for (passNum <- 1 to solverOptions.numPasses) {
        val scalaWeightStr: String = scala.io.Source.fromFile(debugWeightsScalaPattern.format(passNum)).getLines.toList(0)
        val matlabWeightStr: String = scala.io.Source.fromFile(debugWeightsMatlabPattern.format(passNum)).getLines.toList(0)

        val scalaWeightArray: Array[Double] = scalaWeightStr.split(",").map(x => x.toDouble)
        val matlabWeightArray: Array[Double] = matlabWeightStr.split(",").map(x => x.toDouble)

        val zipped = scalaWeightArray zip matlabWeightArray
        val rmse: Double = sqrt(zipped map (sm => abs(sm._1 - sm._2)) reduce ((x, y) => x * x + y * y))

        println("Pass %d: RMSE = %f".format(passNum, rmse))

        assert(rmse < rmseThreshold, "Found RMSE = %f in Pass %d. Threshold is %f".format(rmse, passNum, rmseThreshold))
      }

    }
  }

}

object Main {
  def main(args: Array[String]): Unit = {
    (new BCFWRegression).execute()
    
  }
}
