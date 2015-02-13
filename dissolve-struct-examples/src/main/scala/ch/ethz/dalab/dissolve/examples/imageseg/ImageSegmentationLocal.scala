package ch.ethz.dalab.dissolve.examples.imageseg
import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import breeze.linalg.{ Matrix, Vector, argmax}
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.classification.StructPerceptron
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import ch.ethz.dalab.dissolve.regression.LabeledObject
import breeze.linalg.DenseVector
import breeze.linalg.DenseMatrix

object ImageSegmentationLocal {

  val debugOn = true

  /*
   * read .csv file to an array
   */
  
  def loadData(patternsFilename: String, labelsFilename: String, foldFilename: String): DenseVector[LabeledObject[Matrix[ROIFeature], Matrix[ROILabel]]] = {
    val features: Array[String] = scala.io.Source.fromFile(patternsFilename).getLines().toArray[String]
    val labels: Array[String] = scala.io.Source.fromFile(labelsFilename).getLines().toArray[String]
    val folds: Array[String] = scala.io.Source.fromFile(foldFilename).getLines().toArray[String]
    
    val n = labels.size
    
    assert(features.size == labels.size, "#Features=%d, but #Labels=%d".format(features.size, labels.size))
    assert(features.size == folds.size, "#Features=%d, but #Folds=%d".format(features.size, folds.size))

    val data: DenseVector[LabeledObject[Matrix[ROIFeature], Matrix[ROILabel]]] = DenseVector.fill(n) { null }

    for (i <- 0 until n) {
      // Expected format: id, #rows, #cols, (pixels_i_j)* pixels_n_m
      val feaLine: List[Double] = features(i).split(",").map(x => x.toDouble) toList
      // Expected format: id, #rows, #cols, (pixels_i_j)* pixels_n_m
      val labLine: List[Double] = labels(i).split(",").map(x => x.toDouble) toList
      
//      val feaNumRows: Int = feaLine(1) toInt
//      val feaNumCols: Int = feaLine(2) toInt
//      val labNumRows: Int = labLine(1) toInt
//      val labNumCols: Int = labLine(2) toInt

      val feaNumRows: Int = 16
      val feaNumCols: Int = 8
      val labNumRows: Int = 16
      val labNumCols: Int = 8

      
      assert(feaNumCols == labNumCols, "pattern_i.cols == label_i.cols violated in data")
      assert(feaNumRows == labNumRows, "pattern_i.rows == label_i.rows violated in data")

      val feaVals: Array[ROIFeature] = feaLine.slice(3, 3+128).map(x=>ROIFeature(Vector(x))) toArray
      // The pixel values should be Column-major ordered
      val thisFeature: DenseMatrix[ROIFeature] = DenseVector(feaVals).toDenseMatrix.reshape(feaNumRows, feaNumCols)

      val labVals: Array[ROILabel] = labLine.slice(3, 3+128).map(x=>ROILabel(x.toInt)) toArray
//      assert(List.fromArray(labVals).count(x => x < 0 || x > 26) == 0, "Elements in Labels should be in the range [0, 25]")
      val thisLabel: DenseMatrix[ROILabel] = DenseVector(labVals).toDenseMatrix.reshape(labNumRows, labNumCols)
      
      assert(thisFeature.cols == thisLabel.cols, "pattern_i.cols == label_i.cols violated in Matrix representation")
      assert(thisFeature.rows == thisLabel.rows, "pattern_i.rows == label_i.rows violated in Matrix representation")

      data(i) = new LabeledObject(thisLabel,thisFeature)

    }

    data
  }
  
  /*
  def readCSVLine(src: Source): Array[String] = {
  
    var c = if(src.hasNext) src.next else ' '
    var ret = List[String]()
    var cur = ""
    while (src.hasNext) {
      if (c==',') {
        ret ::= cur
        cur = ""
      }
      else {
        cur += c
      }
    c = src.next
    }
    ret ::= cur
    ret.reverse.toArray
     
    null
  }
  * 
  */
  
  /*
   * load data from feature file
   * "  x y abs         dci         cal
   *  1 1 0.356888964544151 0.913702249526978 0
   *  2 1 0.356888964544151 0.913702249526978 0
   *  3 1 0.359377678682337 0.938085556030273 0
   *  4 1 0.359685008308415 1.00872266292572  0
   *  5 1 0.36041754917366  1.02069628238678  0
   *  6 1 0.358411388626881 0.928287267684937 0"
   */
  
  //def loadData(filePath): Array[String] =  {
  /*
    val src = fromFile("/Users/zhaoyue/git/dissolve-struct/data/image_seg_data/combined_image_with_labels.csv")
    val wholeArray =  readCSVLine(src)
    wholeArray
    * 
    */
  //  null
  //}
  
  
  
  
  /**
   * Feature Function.
   * Uses: http://arxiv.org/pdf/1408.6804v2.pdf
   */
  
  
  
  def featureFn(xMat: Matrix[ROIFeature], yMat: Matrix[ROILabel]): Vector[Double] = {

    assert(xMat.rows == yMat.rows)
    assert(xMat.cols == yMat.cols)

    val x = xMat.toDenseMatrix.toDenseVector
    val y = yMat.toDenseMatrix.toDenseVector

    val numFeatures = x(0).feature.size
    val numClasses = y(0).numClasses
    val numRegions = x.size

    val unary = DenseMatrix.zeros[Double](numFeatures * numRegions, numClasses)
    val pairwise = DenseMatrix.zeros[Double](numClasses, numClasses)

    // Populate the unary features
    for (classNum <- 0 until numClasses) {

      // For each class label, zero-out the x_i whose class label does not agree
      val xTimesIndicator = x.toArray
        .zipWithIndex
        .flatMap {
          case (roiFeature, idx) =>
            if (classNum == y(idx)) // Compare this feature's label
              roiFeature.feature.toArray
            else
              Array.fill(numFeatures)(0.0)
        }

      val startIdx = classNum * numFeatures * numRegions
      val endIdx = (classNum + 1) * numFeatures * numRegions

      unary(::, classNum) := DenseVector(xTimesIndicator)

    }

    // Populate the pairwise features
    for (
      i <- 1 until xMat.rows - 1;
      j <- 1 until xMat.cols - 1
    ) {
      val classA = yMat(i, j).label
      
      for (
        delx <- List(-1, 0, 1);
        dely <- List(-1, 0, 1) if ((delx != 0) && (dely != 0))
      ) {
        val classB = yMat(i + delx, j + dely).label

        pairwise(classA, classB) += 1.0
        pairwise(classB, classA) += 1.0
      }
    }

//    DenseVector.vertcat(unary.toDenseVector, pairwise.toDenseVector)
    unary.toDenseVector
  }

  /**
   * Loss function
   */
  def lossFn(yTruth: Matrix[ROILabel], yPredict: Matrix[ROILabel]): Double = {

    assert(yTruth.rows == yPredict.rows)
    assert(yTruth.cols == yPredict.cols)

    val loss =
      for (
        x <- 0 until yTruth.cols;
        y <- 0 until yTruth.rows
      ) yield {
        if (x == y) 0.0 else 1.0
      }

    loss.sum
  }

  /**
   * Oracle function
   */
  def oracleFn(model: StructSVMModel[Matrix[ROIFeature], Matrix[ROILabel]], xi: Matrix[ROIFeature], yi: Matrix[ROILabel]): Matrix[ROILabel] = {

    assert(xi.rows == yi.rows)
    assert(xi.cols == yi.cols)

    val numClasses = yi(0, 0).numClasses
    val numRows = xi.rows
    val numCols = xi.cols
    val numROI = numRows * numCols
    val xFeatureSize = xi(0, 0).feature.size

    val weightVec = model.getWeights()

    val unaryStartIdx = 0
    val unaryEndIdx = xFeatureSize * numROI * numClasses
    val unary: DenseMatrix[Double] = weightVec(unaryStartIdx until unaryEndIdx)
      .toDenseVector
      .toDenseMatrix
      .reshape(xFeatureSize * numROI, numClasses)

      /*
    val pairwiseStartIdx = unaryEndIdx
    val pairwiseEndIdx = weightVec.size
    assert(pairwiseEndIdx - pairwiseStartIdx == numClasses * numClasses)
    val pairwise: DenseMatrix[Double] = weightVec(pairwiseStartIdx until pairwiseEndIdx)
      .toDenseVector
      .toDenseMatrix
      .reshape(numClasses, numClasses)
      * 
      */


    /*
     * 
    // Declare random variable types
    // A domain and variable type
    object ROIDomain extends DiscreteDomain(numClasses)
    
    // Define a model structure
    
    
    
    
    val feature = featureFn(yi, xi)
    
    val unaryFeature = feature(unaryStartIdx until unaryEndIdx).toDenseMatrix.reshape(xFeatureSize * numROI, numClasses)
    
    val pairwiseFeature = feature(pairStartIdx until pairwiseEndIdx).toDenseMatrix.reshape(numClasses, numClasses)
    
    val scoreFn = unary.t * unaryFeature + pairwise.t * pairwiseFeature
    
    val optimalClass = argmax(scoreFn)._1
    
    val MaximizeByBPChain.
    */
    
    val ret = DenseVector.zeros[Double](xFeatureSize * numROI)
    
    for (
      i<-0 until xFeatureSize*numROI
    ) yield {
      
      ret(i) = argmax(unary(i,::).t)
    }
    ret.map(x=>ROILabel(x.toInt)).toDenseMatrix.reshape(numRows, numCols)
    
    
  }

  /**
   * Prediction Function
   */
  def predictFn(model: StructSVMModel[Matrix[ROIFeature], Matrix[ROILabel]], xi: Matrix[ROIFeature]): Matrix[ROILabel] = {

    val numClasses = 2
    val numRows = xi.rows
    val numCols = xi.cols
    val numROI = numRows * numCols
    val xFeatureSize = xi(0, 0).feature.size

    val weightVec = model.getWeights()

    val unaryStartIdx = 0
    val unaryEndIdx = xFeatureSize * numROI * numClasses
    val unary: DenseMatrix[Double] = weightVec(unaryStartIdx until unaryEndIdx)
      .toDenseVector
      .toDenseMatrix
      .reshape(xFeatureSize * numROI, numClasses)
      
      /*  
    val pairwiseStartIdx = unaryEndIdx
    val pairwiseEndIdx = weightVec.size
    assert(pairwiseEndIdx - pairwiseStartIdx == numClasses * numClasses)
    val pairwise: DenseMatrix[Double] = weightVec(pairwiseStartIdx until pairwiseEndIdx)
      .toDenseVector
      .toDenseMatrix
      .reshape(numClasses, numClasses)
      * 
      */
      
    val ret = DenseVector.zeros[Double](xFeatureSize * numROI)
    
    for (
      i<-0 until xFeatureSize*numROI
    ) yield {
      ret(i) = argmax(unary(i,::).t)
    }
    ret.map(x=>ROILabel(x.toInt)).toDenseMatrix.reshape(numRows, numCols)
  }

  def dissolveImageSegmentation(options: Map[String, String]) {
    
    /**
     * Load all options
     */
    val PERC_TRAIN: Double = options.getOrElse("perctrain", "0.05").toDouble // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)
    
    val appName: String = options.getOrElse("appname", "Chain-Dissolve")
    
    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")
    
    val runLocally: Boolean = options.getOrElse("local", "true").toBoolean

    val solverOptions: SolverOptions[Matrix[ROIFeature], Matrix[ROILabel]] = new SolverOptions()
    solverOptions.roundLimit = options.getOrElse("numpasses", "5").toInt // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean
    solverOptions.lambda = options.getOrElse("lambda", "0.01").toDouble
    solverOptions.doWeightedAveraging = options.getOrElse("wavg", "false").toBoolean
    solverOptions.doLineSearch = options.getOrElse("linesearch", "true").toBoolean
    solverOptions.debug = options.getOrElse("debugloss", "false").toBoolean

    solverOptions.sample = options.getOrElse("sample", "frac")
    solverOptions.sampleFrac = options.getOrElse("samplefrac", "0.5").toDouble
    solverOptions.sampleWithReplacement = options.getOrElse("samplewithreplacement", "false").toBoolean
    
    solverOptions.enableManualPartitionSize = options.getOrElse("manualrddpart", "false").toBoolean
    solverOptions.NUM_PART = options.getOrElse("numpart", "2").toInt
    
    solverOptions.enableOracleCache = options.getOrElse("enableoracle", "false").toBoolean
    solverOptions.oracleCacheSize = options.getOrElse("oraclesize", "5").toInt
    
    solverOptions.debugInfoPath = options.getOrElse("debugpath", debugDir + "/debug-dissolve-%d.csv".format(System.currentTimeMillis()))
    
    /**
     * Some local overrides
     */
    if(runLocally) {
      solverOptions.sampleFrac = 1.0
      solverOptions.enableOracleCache = false
      solverOptions.oracleCacheSize = 10
      solverOptions.roundLimit = 5
      solverOptions.enableManualPartitionSize = true
      solverOptions.NUM_PART = 1
      solverOptions.doWeightedAveraging = false
    }

    println(solverOptions.toString())

    /**
     * Begin execution
     */
    
    val trainDataUnord: Vector[LabeledObject[Matrix[ROIFeature], Matrix[ROILabel]]] = loadData(dataDir + "/patterns_train.csv", dataDir + "/patterns_train.csv", dataDir + "/folds_train.csv")
    val testDataUnord: Vector[LabeledObject[Matrix[ROIFeature], Matrix[ROILabel]]] = loadData(dataDir + "/patterns_test.csv", dataDir + "/patterns_test.csv", dataDir + "/folds_test.csv")

    println(" Loaded data with %d rows, feature=%dx%d, label=%dx%d".format(trainDataUnord.size, trainDataUnord(0).pattern.rows, trainDataUnord(0).pattern.cols, trainDataUnord(0).label.rows, trainDataUnord(0).label.cols))

    val conf = 
      if(runLocally)
        new SparkConf().setAppName(appName).setMaster("local")
      else
        new SparkConf().setAppName(appName)
        
    val sc = new SparkContext(conf)
    sc.setCheckpointDir(debugDir + "/checkpoint-files")
    
    println(SolverUtils.getSparkConfString(sc.getConf))

    // Read order from the file and permute the Vector accordingly
    val trainOrder: String = "../data/perm_train.csv"
    val permLine: Array[String] = scala.io.Source.fromFile(trainOrder).getLines().toArray[String]
    assert(permLine.size == 1)
    val perm = permLine(0).split(",").map(x => x.toInt - 1) // Reduce by 1 because of order is Matlab indexed
    val train_data: Array[LabeledObject[Matrix[ROIFeature], Matrix[ROILabel]]] = trainDataUnord(List.fromArray(perm).slice(0, (PERC_TRAIN * trainDataUnord.size).toInt)).toArray
    
    
    solverOptions.testDataRDD = 
      if(solverOptions.enableManualPartitionSize)
        Some(sc.parallelize(testDataUnord.toArray, solverOptions.NUM_PART))
      else
        Some(sc.parallelize(testDataUnord.toArray))
        
    val trainDataRDD = 
      if(solverOptions.enableManualPartitionSize)
        sc.parallelize(train_data, solverOptions.NUM_PART)
      else
        sc.parallelize(train_data)
    val trainer: StructPerceptron[Matrix[ROIFeature], Matrix[ROILabel]] = new StructPerceptron[Matrix[ROIFeature], Matrix[ROILabel]](
      train_data,
      featureFn,
      lossFn,
      oracleFn,
      predictFn,
      solverOptions)

    val model: StructSVMModel[Matrix[ROIFeature], Matrix[ROILabel]] = trainer.trainModel()

    var avgTrainLoss: Double = 0.0
    for (item <- train_data) {
      val prediction = model.predictFn(model, item.pattern)
      avgTrainLoss += lossFn(item.label, prediction)
    }
    println("Average loss on training set = %f".format(avgTrainLoss / train_data.size))

    var avgTestLoss: Double = 0.0
    for (item <- testDataUnord) {
      val prediction = model.predictFn(model, item.pattern)
      avgTestLoss += lossFn(item.label, prediction)
    }
    println("Average loss on test set = %f".format(avgTestLoss / testDataUnord.size))
   
  }
  
  

  def main(args: Array[String]): Unit = {
    PropertyConfigurator.configure("conf/log4j.properties")

    val options: Map[String, String] = args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt)    => (opt -> "true")
        case _             => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    System.setProperty("spark.akka.frameSize", "512")
    println(options)

    dissolveImageSegmentation(options)

  }

}
