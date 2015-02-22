package ch.ethz.dalab.dissolve.examples.imageseg

import scala.annotation.elidable
import scala.annotation.elidable.ASSERTION
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable.Buffer
import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import breeze.linalg.Axis
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import breeze.linalg.Vector
import breeze.linalg.max
import breeze.linalg.sum
import breeze.numerics.abs
import cc.factorie.infer.SamplingMaximizer
import cc.factorie.infer.VariableSettingsSampler
import cc.factorie.model.{ Factor, Factor1, Factor2 }
import cc.factorie.model.CombinedModel
import cc.factorie.model.ItemizedModel
import cc.factorie.model.TupleTemplateWithStatistics2
import cc.factorie.singleFactorIterable
import cc.factorie.variable.DiscreteDomain
import cc.factorie.variable.DiscreteVariable
import cc.factorie.variable.IntegerVariable
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.optimization.SolverOptions
import org.apache.spark.SparkContext
import ch.ethz.dalab.dissolve.optimization.SolverUtils
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
import scala.io.Source
import cc.factorie.infer.MaximizeByMPLP

case class ROIFeature(feature: Vector[Double], name: String = "NA") // Represent each pixel/region by a feature vector

case class ROILabel(label: Int, numClasses: Int = 24) {
  override def equals(o: Any) = o match {
    case that: ROILabel => that.label == this.label
    case _              => false
  }
}

object ImageSegmentationDemo {

  /**
   * Given:
   * - a matrix xMat of super-pixels, of size r = n x m, and x_i, an f-dimensional vector
   * - corresponding labels of these super-pixels yMat, with K classes
   * Return:
   * - a matrix of size (f*K) x r, i.e, each column corresponds to the feature map of x_i
   */
  def getUnaryFeatureMap(yMat: DenseMatrix[ROILabel], xMat: DenseMatrix[ROIFeature]): DenseMatrix[Double] = {
    assert(xMat.rows == yMat.rows)
    assert(xMat.cols == yMat.cols)

    // println("xMat.size = %d x %d".format(xMat.rows, xMat.cols))

    val numFeatures = xMat(0, 0).feature.size // f
    val numClasses = yMat(0, 0).numClasses // K
    val numRegions = xMat.rows * xMat.cols // r

    val unaryMat = DenseMatrix.zeros[Double](numFeatures * numClasses, numRegions)

    /**
     * Populate unary features
     * For each node i in graph defined by xMat, whose feature vector is x_i and corresponding label is y_i,
     * construct a feature map phi_i given by: [I(y_i = 0)x_i I(y_i = 1)x_i ... I(y_i = K)x_i ]
     */

    xMat.keysIterator.foreach {
      case (r, c) =>
        val i = columnMajorIdx(r, c, xMat.rows) // Column-major iteration

        // println("(r, c) = (%d, %d)".format(r, c))
        val x_i = xMat(r, c).feature
        val y_i = yMat(r, c).label

        val phi_i = DenseVector.zeros[Double](numFeatures * numClasses)

        val startIdx = numFeatures * y_i
        val endIdx = startIdx + numFeatures

        // For y_i'th position of phi_i, populate x_i's feature vector
        /*println("x_i.size = " + x_i.size)
        println("endIdx - startIdx = " + (endIdx - startIdx))
        println("startIdx = %d\tendIdx = %d".format(startIdx, endIdx))
        println("phi_i.size = " + phi_i.size)*/

        phi_i(startIdx until endIdx) := x_i

        unaryMat(::, i) := phi_i
    }

    unaryMat
  }

  def columnMajorIdx(r: Int, c: Int, numRows: Int) = c * numRows + r

  /**
   * Given:
   * - a matrix xMat of super-pixels, of size r = n x m, and x_i, an f-dimensional vector
   * - corresponding labels of these super-pixels yMat, with K classes
   * Return:
   * - a matrix of size f x r, each column corresponding to a (histogram) feature vector of region r
   */
  def getUnaryFeatureMap(xMat: DenseMatrix[ROIFeature]): DenseMatrix[Double] = {
    // println("xMat.size = %d x %d".format(xMat.rows, xMat.cols))

    val numDims = xMat(0, 0).feature.size // f
    val numClasses = 24
    val numRegions = xMat.rows * xMat.cols // r

    val unaryMat = DenseMatrix.zeros[Double](numDims, numRegions)

    /**
     * Populate unary features
     * For each node i in graph defined by xMat, whose feature vector is x_i and corresponding label is y_i,
     * construct a feature map phi_i given by: [I(y_i = 0)x_i I(y_i = 1)x_i ... I(y_i = K)x_i ]
     */

    xMat.keysIterator
      .foreach {
        case (r, c) =>
          val idx = columnMajorIdx(r, c, xMat.rows)
          unaryMat(::, idx) := xMat(r, c).feature
      }

    unaryMat
  }

  /**
   * Given:
   * - a matrix xMat of super-pixels, of size r = n x m, and x_i, an f-dimensional vector
   * - corresponding labels of these super-pixels yMat, with K classes
   * return:
   * - a matrix of size K x K, capturing pairwise scores
   */
  def getPairwiseFeatureMap(yMat: DenseMatrix[ROILabel], xMat: DenseMatrix[ROIFeature]): DenseMatrix[Double] = {
    assert(xMat.rows == yMat.rows)
    assert(xMat.cols == yMat.cols)

    val numFeatures = xMat(0, 0).feature.size
    val numClasses = yMat(0, 0).numClasses
    val numRegions = xMat.rows * xMat.cols

    val pairwiseMat = DenseMatrix.zeros[Double](numClasses, numClasses)

    /*for (
      c <- 1 until xMat.cols - 1;
      r <- 1 until xMat.rows - 1
    ) {
      val classA = yMat(r, c).label

      // Iterate through all neighbours
      for (
        delx <- List(-1, 0, 1);
        dely <- List(-1, 0, 1) if ((delx != 0) && (dely != 0))
      ) {
        val classB = yMat(r + delx, c + dely).label

        pairwiseMat(classA, classB) += 1.0
        pairwiseMat(classB, classA) += 1.0
      }
    }*/

    for (
      y <- 0 until xMat.cols;
      x <- 0 until xMat.rows
    ) {
      val classA = yMat(x, y).label

      val neighbours = List((-1, 0), (0, -1), (1, 0), (0, 1))

      for ((dx, dy) <- neighbours if (x + dx >= 0) && (y + dy >= 0) && (x + dx < xMat.rows) && (y + dy < xMat.cols)) {
        val classB = yMat(x + dx, y + dy).label
        pairwiseMat(classA, classB) += 1.0
        pairwiseMat(classB, classA) += 1.0
      }
    }

    /*println("Pairwise feature map - " + xMat(0, 0).name)
    println(pairwiseMat(2 to 4, 2 to 4))*/

    pairwiseMat
  }

  /**
   * Feature Function.
   * Uses: http://arxiv.org/pdf/1408.6804v2.pdf
   * http://www.kev-smith.com/papers/LUCCHI_ECCV12.pdf
   */
  def featureFn(xMat: DenseMatrix[ROIFeature], yMat: DenseMatrix[ROILabel]): Vector[Double] = {

    assert(xMat.rows == yMat.rows)
    assert(xMat.cols == yMat.cols)

    // val unaryMat = getUnaryFeatureMap(yMat, xMat)
    // val pairwiseMat = getPairwiseFeatureMap(yMat, xMat)

    val numDims = xMat(0, 0).feature.size
    val numClasses = yMat(0, 0).numClasses

    val phi = DenseVector.zeros[Double](numDims * numClasses + numClasses * numClasses)

    // Unaries vector, of size f * K. Each column corresponds to a feature vector
    val unary = DenseVector.zeros[Double](numDims * numClasses)
    yMat.keysIterator.foreach {
      case (r, c) =>
        val label = yMat(r, c).label
        val startIdx = label * numClasses
        val endIdx = startIdx + numDims
        unary(startIdx until endIdx) := xMat(r, c).feature + unary(startIdx until endIdx)
    }

    val pairwise = getPairwiseFeatureMap(yMat, xMat).toDenseVector

    phi(0 until (numDims * numClasses)) := unary
    phi((numDims * numClasses) until phi.size) := pairwise

    // println("numDims = %d\tnumClasses = %d\tphi.size = %d".format(numDims, numClasses, phi.size))

    phi
  }

  /**
   * Loss function
   */
  def lossFn(yTruth: DenseMatrix[ROILabel], yPredict: DenseMatrix[ROILabel]): Double = {

    assert(yTruth.rows == yPredict.rows)
    assert(yTruth.cols == yPredict.cols)

    val loss =
      for (
        y <- 0 until yTruth.cols;
        x <- 0 until yTruth.rows
      ) yield {
        if (yTruth(x, y).label == yPredict(x, y).label) 0.0 else 1.0
      }

    loss.sum / (yTruth.cols * yTruth.rows)
  }

  /**
   * Takes as input:
   * - a weight vector of size: (f * K) + (K * K), i.e, unary features [xFeatureSize * numClasses] + pairwise features [numClasses * numClasses]
   * - feature size of x
   * - number of class labels
   * - padded - if padded, returns [I(K=0) w_0 ... I(K=0) w_k], else returns a matrix of size f x K: [w_1 w_2 ... w_k], i.e, each column is feature representation of some label
   */
  def unpackWeightVec(weightVec: Vector[Double], xFeatureSize: Int, numClasses: Int = 24, padded: Boolean = false): (DenseMatrix[Double], DenseMatrix[Double]) = {
    // Unary features
    val startIdx = 0
    val endIdx = xFeatureSize * numClasses
    val unaryFeatureVec = weightVec(startIdx until endIdx).toDenseVector // Stored as [|--f(k=0)--||--f(k=1)--| ... |--f(K=k)--|]
    val tempUnaryPot = unaryFeatureVec.toDenseMatrix.reshape(xFeatureSize, numClasses)

    val unaryPot =
      if (padded) {
        // Each column in this vector contains [I(K=0) w_0 ... I(K=0) w_k]
        val unaryPotPadded = DenseMatrix.zeros[Double](xFeatureSize * numClasses, numClasses)
        for (k <- 0 until numClasses) {
          val w = tempUnaryPot(::, k)
          val startIdx = k * xFeatureSize
          val endIdx = startIdx + xFeatureSize
          unaryPotPadded(startIdx until endIdx, k) := w
        }
        unaryPotPadded
      } else {
        tempUnaryPot
      }

    // Pairwise feature Vector
    val pairwiseFeatureVec = weightVec(endIdx until weightVec.size).toDenseVector
    assert(pairwiseFeatureVec.size == numClasses * numClasses)
    val pairwisePot = pairwiseFeatureVec.toDenseMatrix.reshape(numClasses, numClasses)

    (unaryPot, pairwisePot)
  }

  /**
   * thetaUnary is of size r x K, where is the number of regions
   * thetaPairwise is of size K x K
   */
  def decodeFn(thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], imageWidth: Int, imageHeight: Int): DenseMatrix[ROILabel] = {

    println("thetaUnary(3, ::): \n" + thetaUnary(3, ::))
    println("thetaPairwise(2 to 4, 2 to 4): ): \n" + thetaPairwise(2 to 4, 2 to 4))

    /**
     *  Construct a model such that there exists 2 kinds of variables - Region and Pixel
     *  Region encodes the fixed (x, y) position of the Pixel in the image
     *  Pixel encodes the possible label
     *
     *  Factors are between a Region and Pixel, scores given by - thetaUnary
     *  and between two Pixels, scores given by - thetaPairwise
     */

    val numRegions: Int = thetaUnary.rows
    val numClasses: Int = thetaUnary.cols

    assert(thetaPairwise.rows == numClasses)

    class RegionVar(val score: Int) extends IntegerVariable(score)

    object PixelDomain extends DiscreteDomain(numClasses)

    class Pixel(val x: Int, val y: Int, val image: Seq[Seq[Pixel]], val truth: Int) extends DiscreteVariable(truth) {

      def domain = PixelDomain

      val region: RegionVar = new RegionVar(columnMajorIdx(x, y, image.size))
    }

    object LocalTemplate extends TupleTemplateWithStatistics2[Pixel, RegionVar] {

      val alpha = 1.0
      def score(k: Pixel#Value, r: RegionVar#Value) = {
        // if (r.intValue == 3) println("thetaUnary(%d, %d) = %f".format(r.intValue, k.intValue, thetaUnary(r.intValue, k.intValue)))
        thetaUnary(r.intValue, k.intValue)
      }
      def unroll1(p: Pixel) = Factor(p, p.region)
      def unroll2(r: RegionVar) = Nil
    }

    object PairwiseTemplate extends TupleTemplateWithStatistics2[Pixel, Pixel] {

      def score(v1: Pixel#Value, v2: Pixel#Value) = if (v1.intValue == v2.intValue) 1.0 else -1.0

      def unroll1(v: Pixel) = {
        // println("v.x = %d, v.y = %d".format(v.x, v.y))
        val img = v.image
        val factors = new ArrayBuffer[FactorType]
        if (v.x < img.length - 1) factors += Factor(v, img(v.x + 1)(v.y))
        if (v.y < img(0).length - 1) factors += Factor(v, img(v.x)(v.y + 1))
        if (v.x > 0) factors += Factor(img(v.x - 1)(v.y), v)
        if (v.y > 0) factors += Factor(img(v.x)(v.y - 1), v)
        factors
      }

      def unroll2(v2: Pixel) = Nil
    }

    // Convert the image into a grid of Factorie variables
    val image: Buffer[Seq[Pixel]] = new ArrayBuffer
    for (i <- 0 until imageHeight) {
      val row = new ArrayBuffer[Pixel]

      for (j <- 0 until imageWidth) {
        row += new Pixel(i, j, image, 0)
      }

      image += row
    }

    // println("image rows = %d, cols = %d".format(image.size, image(0).size))

    // Run MAP inference
    val pixels = image.flatMap(_.toSeq).toSeq
    // val maxx = pixels.map { x => x.x }.reduce((x, y) => max(x, y))
    // val maxy = pixels.map { y => y.y }.reduce((x, y) => max(x, y))
    // val maxr = pixels.map { p => p.region.intValue }.reduce((x, y) => max(x, y))
    // println("pixels: max x = %d, max y = %d, max r = %d".format(maxx, maxy, maxr))

    val gridModel = new CombinedModel(LocalTemplate, PairwiseTemplate)
    implicit val random = new scala.util.Random(0)
    pixels.foreach(_.set(0)(null))

    val sampler = new SamplingMaximizer[Pixel](new VariableSettingsSampler(gridModel))
    val foo = sampler.maximize(pixels)

    // pixels.foreach(_.set(1)(null))
    // println(pixels.map { x => x.intValue })

    // Retrieve assigned labels from these pixels
    val imgMask: DenseMatrix[ROILabel] = new DenseMatrix[ROILabel](imageHeight, imageWidth)
    for (i <- 0 until imageHeight) {
      for (j <- 0 until imageWidth) {
        imgMask(i, j) = ROILabel(image(i)(j).intValue)
      }
    }

    ImageSegmentationUtils.printLabeledImage(imgMask)

    imgMask
  }

  /**
   * thetaUnary is of size r x K, where is the number of regions
   * thetaPairwise is of size K x K
   */
  def decodeFnAlt(thetaUnary: DenseMatrix[Double], thetaPairwise: DenseMatrix[Double], imageWidth: Int, imageHeight: Int): DenseMatrix[ROILabel] = {

    val numRegions: Int = thetaUnary.rows
    val numClasses: Int = thetaUnary.cols

    assert(thetaPairwise.rows == numClasses)

    // Convert the image into a grid of Factorie variables
    val image: Buffer[Seq[Pixel]] = new ArrayBuffer
    for (i <- 0 until imageHeight) {
      val row = new ArrayBuffer[Pixel]

      for (j <- 0 until imageWidth) {
        row += new Pixel(0)
      }

      image += row
    }

    class RegionVar(val score: Int) extends IntegerVariable(score)

    object PixelDomain extends DiscreteDomain(numClasses)

    class Pixel(i: Int) extends DiscreteVariable(i) {
      def domain = PixelDomain
    }

    def getUnaryFactor(yi: Pixel, x: Int, y: Int): Factor = {
      new Factor1(yi) {
        val r = columnMajorIdx(x, y, image.size)
        def score(k: Pixel#Value) = thetaUnary(r, k.intValue)
      }
    }

    def getPairwiseFactor(yi: Pixel, yj: Pixel): Factor = {
      new Factor2(yi, yj) {
        def score(i: Pixel#Value, j: Pixel#Value) = thetaPairwise(i.intValue, j.intValue)
      }
    }

    val indexedPixels =
      for {
        x <- 0 until image.size;
        y <- 0 until image(0).size
      } yield {
        ((x, y), image(x)(y))
      }
    val regionIndexedPixels =
      indexedPixels.map {
        case ((x, y), pix) =>
          val r = columnMajorIdx(x, y, image.size)
          (r, pix)
      }

    val pixels: IndexedSeq[Pixel] = indexedPixels.map(_._2)

    val unaries: IndexedSeq[Factor] = indexedPixels.map {
      case ((x, y), pix) =>
        getUnaryFactor(pix, x, y)
    }

    val pairwise: IndexedSeq[Factor] =
      indexedPixels.flatMap {
        case ((x, y), pix) =>
          val factors = new ArrayBuffer[Factor]

          // (x, y) and (x, y+1)
          if (y < imageWidth - 1)
            factors ++= getPairwiseFactor(pix, image(x)(y + 1))

          // (x, y) and (x+1, y)
          if (x < imageHeight - 1)
            factors ++= getPairwiseFactor(pix, image(x + 1)(y))

          // (x, y) and (x+1, y+1)
          if ((y < imageWidth - 1) && (x < imageHeight - 1))
            factors ++= getPairwiseFactor(pix, image(x + 1)(y + 1))

          // (x, y) and (x, y-1)
          // if (y > 0)
          // factors ++= getPairwiseFactor(pix, image(x)(y - 1))

          // (x, y) and (x-1, y)
          // if (x > 0)
          // factors ++= getPairwiseFactor(pix, image(x - 1)(y))

          factors
      }

    val model = new ItemizedModel
    model ++= unaries
    model ++= pairwise

    val assgn = MaximizeByMPLP.infer(pixels, model).mapAssignment

    // Retrieve assigned labels from these pixels
    val imgMask: DenseMatrix[ROILabel] = new DenseMatrix[ROILabel](imageHeight, imageWidth)
    for (i <- 0 until imageHeight) {
      for (j <- 0 until imageWidth) {
        imgMask(i, j) = ROILabel(assgn(image(i)(j)).intValue)
      }
    }

    // ImageSegmentationUtils.printLabeledImage(imgMask)

    imgMask
  }

  /**
   * Oracle function
   */
  def oracleFn(model: StructSVMModel[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]], xi: DenseMatrix[ROIFeature], yi: DenseMatrix[ROILabel]): DenseMatrix[ROILabel] = {

    // assert(xi.rows == yi.rows)
    // assert(xi.cols == yi.cols)

    val numClasses = 24
    val numRows = xi.rows
    val numCols = xi.cols
    val numROI = numRows * numCols
    val numDims = xi(0, 0).feature.size

    val weightVec = model.getWeights()

    /*
    // Unary is of size (f*K) x K
    // Pairwise is of size K x K
    val (unary, pairwise) = unpackWeightVec(weightVec, xFeatureSize, numClasses)
    assert(unary.rows == numClasses * xFeatureSize)
    assert(unary.cols == pairwise.cols)
    */

    // Unary is of size f x K, each column representing feature vector of class K
    // Pairwise if of size K * K
    val (unaryWeights, pairwiseWeights) = unpackWeightVec(weightVec, numDims, numClasses = numClasses, padded = false)
    assert(unaryWeights.rows == numDims)
    assert(unaryWeights.cols == numClasses)
    assert(pairwiseWeights.rows == pairwiseWeights.cols)
    assert(pairwiseWeights.rows == numClasses)

    val phi_Y: DenseMatrix[Double] = getUnaryFeatureMap(xi) // Retrieves a f x r matrix representation of the original image, i.e, each column is the feature vector that region r
    val thetaUnary = phi_Y.t * unaryWeights // Returns a r x K matrix, where theta(r, k) is the unary potential of region i having label k
    val thetaPairwise = pairwiseWeights

    // If yi is present, do loss-augmentation
    if (yi != null) {
      yi.keysIterator.foreach {
        case (r, c) =>
          val idx = columnMajorIdx(r, c, yi.rows)
          val yLab = yi(r, c)
          thetaUnary(idx, ::) := thetaUnary(idx, ::) + 1.0 / numROI
          // Loss augmentation step
          val k = yLab.label
          thetaUnary(idx, k) = thetaUnary(idx, k) - 1.0 / numROI
      }

    }

    // println(thetaUnary(0 until 5, 0 until 3).toDenseVector)
    val eps = 0.000001
    /*println("Decoding: " + xi(0, 0).name)
    print("weightVec: ")
    println(weightVec.toDenseVector.keysIterator.map(idx => (idx, weightVec(idx))).filter(x => abs(x._2) > eps).map(x => "(%d, %f)".format(x._1, x._2)).toList.take(20))
    print("xi(0, 0): ")
    println(xi(0, 0).feature.keysIterator.map(idx => (idx, xi(0, 0).feature(idx))).filter(x => abs(x._2) > eps).map(x => "(%d, %f)".format(x._1, x._2)).toList.take(20))
    print("phi_Y: ")
    println(phi_Y.activeKeysIterator.map { case (r, c) => (r, c, phi_Y(r, c)) }.filter(x => abs(x._3) > eps).filter(_._2 == 3).map(x => "(%d, %d, %f)".format(x._1, x._2, x._3)).toList.take(20))
    print("unaryWeights: ")
    println(unaryWeights.activeKeysIterator.map { case (r, c) => (r, c, unaryWeights(r, c)) }.filter(x => abs(x._3) > eps).map(x => "(%d, %d, %f)".format(x._1, x._2, x._3)).toList.take(20))
    print("thetaUnary: ")
    println(thetaUnary.activeKeysIterator.map { case (r, c) => (r, c, thetaUnary(r, c)) }.filter(x => abs(x._3) > eps).map(x => "(%d, %d, %f)".format(x._1, x._2, x._3)).toList.take(20))
    print("thetaPairwise: ")
    println(thetaPairwise.activeKeysIterator.map { case (r, c) => (r, c, thetaPairwise(r, c)) }.filter(x => abs(x._3) > eps).map(x => "(%d, %d, %f)".format(x._1, x._2, x._3)).toList.take(20))
    // println("thetaPairwise(2:4, 2:4): \n" + thetaPairwise(2 to 4, 2 to 4))
    println()*/

    // println("unary.size = %d x %d".format(unary.rows, unary.cols))
    // println("theta_unary.size = %d x %d".format(thetaUnary.rows, thetaUnary.cols))
    // println("theta_pairwise.size = %d x %d".format(thetaPairwise.rows, thetaPairwise.cols))

    /**
     * Parameter estimation
     */

    decodeFnAlt(thetaUnary, thetaPairwise, numCols, numRows)
  }

  /**
   * Prediction Function
   */
  def predictFn(model: StructSVMModel[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]], xi: DenseMatrix[ROIFeature]): DenseMatrix[ROILabel] = {
    oracleFn(model, xi, null)
  }

  def dissolveImageSementation(options: Map[String, String]) {

    val PERC_TRAIN: Double = 0.05 // Restrict to using a fraction of data for training (Used to overcome OutOfMemory exceptions while testing locally)

    val msrcDir: String = "../data/generated"

    val appName: String = options.getOrElse("appname", "ImageSeg")

    val dataDir: String = options.getOrElse("datadir", "../data/generated")
    val debugDir: String = options.getOrElse("debugdir", "../debug")

    val runLocally: Boolean = options.getOrElse("local", "true").toBoolean

    val solverOptions: SolverOptions[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]] = new SolverOptions()
    solverOptions.roundLimit = options.getOrElse("roundLimit", "5").toInt // After these many passes, each slice of the RDD returns a trained model
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean
    solverOptions.lambda = options.getOrElse("lambda", "0.01").toDouble
    solverOptions.doWeightedAveraging = options.getOrElse("wavg", "false").toBoolean
    solverOptions.doLineSearch = options.getOrElse("linesearch", "true").toBoolean
    solverOptions.debug = options.getOrElse("debug", "false").toBoolean

    solverOptions.sample = options.getOrElse("sample", "frac")
    solverOptions.sampleFrac = options.getOrElse("samplefrac", "0.5").toDouble
    solverOptions.sampleWithReplacement = options.getOrElse("samplewithreplacement", "false").toBoolean

    solverOptions.enableManualPartitionSize = options.getOrElse("manualrddpart", "false").toBoolean
    solverOptions.NUM_PART = options.getOrElse("numpart", "2").toInt

    solverOptions.enableOracleCache = options.getOrElse("enableoracle", "false").toBoolean
    solverOptions.oracleCacheSize = options.getOrElse("oraclesize", "5").toInt

    solverOptions.debugInfoPath = options.getOrElse("debugpath", debugDir + "/imageseg-%d.csv".format(System.currentTimeMillis()))

    /**
     * Some local overrides
     */
    if (runLocally) {
      solverOptions.sampleFrac = 1.0
      solverOptions.enableOracleCache = false
      solverOptions.oracleCacheSize = 10
      solverOptions.stoppingCriterion = solverOptions.RoundLimitCriterion
      solverOptions.roundLimit = 10
      solverOptions.enableManualPartitionSize = true
      solverOptions.NUM_PART = 1
      solverOptions.doWeightedAveraging = false

      solverOptions.debug = false
      solverOptions.debugMultiplier = 1
    }

    println(solverOptions.toString())

    val (trainData, testData) = ImageSegmentationUtils.loadMSRC("../data/generated/MSRC_ObjCategImageDatabase_v2")

    println("Running Distributed BCFW with CoCoA. Loaded data with %d rows, pattern=%dx%d, label=%dx1".format(trainData.size, trainData(0).pattern.rows, trainData(0).pattern.cols, trainData(0).label.size))

    val conf =
      if (runLocally)
        new SparkConf().setAppName(appName).setMaster("local")
      else
        new SparkConf().setAppName(appName)

    val sc = new SparkContext(conf)
    sc.setCheckpointDir(debugDir + "/checkpoint-files")

    println(SolverUtils.getSparkConfString(sc.getConf))

    solverOptions.testDataRDD =
      if (solverOptions.enableManualPartitionSize)
        Some(sc.parallelize(testData, solverOptions.NUM_PART))
      else
        Some(sc.parallelize(testData))

    val trainDataRDD =
      if (solverOptions.enableManualPartitionSize)
        sc.parallelize(trainData, solverOptions.NUM_PART)
      else
        sc.parallelize(trainData)

    val trainer: StructSVMWithDBCFW[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]] =
      new StructSVMWithDBCFW[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]](
        trainDataRDD,
        featureFn,
        lossFn,
        oracleFn,
        predictFn,
        solverOptions)

    val model: StructSVMModel[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]] = trainer.trainModel()

    var avgTrainLoss: Double = 0.0
    val trainingFileNames = Source.fromURL(getClass.getResource("/imageseg_cattle_train.txt")).getLines().toArray
    trainData.zip(trainingFileNames).foreach {
      case (item, fname) =>
        val prediction = model.predictFn(model, item.pattern)
        avgTrainLoss += lossFn(item.label, prediction)

        val outname = fname.split('.')(0)
        ImageSegmentationUtils.printLabeledImage(item.label, "%s/%s.jpg".format(debugDir, outname))
        ImageSegmentationUtils.printLabeledImage(prediction, "%s/%s_p.jpg".format(debugDir, outname))

    }
    println("Average loss on training set = %f".format(avgTrainLoss / trainData.size))

    var avgTestLoss: Double = 0.0
    for (item <- testData) {
      val prediction = model.predictFn(model, item.pattern)
      avgTestLoss += lossFn(item.label, prediction)
    }
    println("Average loss on test set = %f".format(avgTestLoss / testData.size))

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

    dissolveImageSementation(options)

  }

}