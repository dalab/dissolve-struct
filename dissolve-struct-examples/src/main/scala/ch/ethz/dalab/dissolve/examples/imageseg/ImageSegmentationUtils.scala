package ch.ethz.dalab.dissolve.examples.imageseg

import java.io.File
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import breeze.linalg.{ Matrix, Vector }
import ch.ethz.dalab.dissolve.regression.LabeledObject
import scala.io.Source
import breeze.linalg.DenseMatrix
import java.awt.image.DataBufferInt
import breeze.linalg.DenseVector
import breeze.linalg.max
import breeze.linalg.min
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import java.io.PrintWriter
import java.awt.image.DataBufferByte

object ImageSegmentationUtils {
  val REGION_WIDTH = 10
  val REGION_HEIGHT = 10

  val featurizer_options: List[String] = List("HIST")

  val colormapFile = "/imageseg_colormap.txt"
  val colormap: Map[Int, Int] = Source.fromURL(getClass.getResource(colormapFile))
    .getLines()
    .map { line => line.split(" ") }
    .map {
      case Array(label, value, r, g, b, className) =>
        value.toInt -> label.toInt
    }
    .toMap

  val labelToRGB: Map[Int, Int] = Source.fromURL(getClass.getResource(colormapFile))
    .getLines()
    .map { line => line.split(" ") }
    .map {
      case Array(label, value, r, g, b, className) =>
        val rgb = ((r.toInt & 0x0ff) << 16) |
          ((g.toInt & 0x0ff) << 8) |
          (b.toInt & 0x0ff)
        label.toInt -> rgb
    }
    .toMap

  def printLabeledImage(img: DenseMatrix[ROILabel], outputFile: String): Unit = {

    val out: BufferedImage = new BufferedImage(img.cols, img.rows, BufferedImage.TYPE_INT_RGB)

    img.keysIterator.foreach {
      case (r, c) =>
        out.setRGB(c, r, labelToRGB(img(r, c).label))
    }

    // val img: BufferedImage = ImageIO.read(new File(imgPath))
    ImageIO.write(out, "jpg", new File(outputFile))

  }

  def printLabeledImage(img: DenseMatrix[ROILabel]): Unit = {

    for (r <- 0 until img.rows) {
      for (c <- 0 until img.cols) {
        print("%2d ".format(img(r, c).label))
      }
      println()
    }

  }

  /**
   * Constructs a histogram vector using pixel (i, j) and a surrounding region of size (width x height)
   */
  def histogramFeaturizer(patch: BufferedImage, patchWithContext: BufferedImage): ROIFeature = {

    // The intensities are split into these many bins.
    // For example, in case of 4 bins, Bin 0 corresponds to intensities 0-63, bin 1 is 64-127,
    // bin 2 is 128-191, and bin 3 is 192-255.
    val NUM_BINS = 4
    // Store the histogram in 3 blocks of [ R | G | B ]
    // The index in the histogram feature vector is function of R,G,B intensity bins
    val histogramVector = DenseVector.zeros[Double](NUM_BINS * NUM_BINS * NUM_BINS)

    // Convert patchWithContext region into an ARGB vector
    /*val imageRGBVector: Array[Int] = patch.getRaster()
      .getDataBuffer().asInstanceOf[DataBufferInt]
      .getData()*/

    val imageRGBVector = patch.getRGB(0, 0, patch.getWidth, patch.getHeight, null, 0, patch.getWidth)

    /*val rgb = patch.getRGB(0, 0)
    val red = (rgb >> 16) & 0xFF
    val green = (rgb >> 8) & 0xFF
    val blue = (rgb) & 0xFF
    println("(%4d,%4d,%4d)".format(red, green, blue))

    val rgb2 = imageRGBVector(0)
    val red2 = (rgb2 >> 16) & 0xFF
    val green2 = (rgb2 >> 8) & 0xFF
    val blue2 = (rgb2) & 0xFF
    println("(%4d,%4d,%4d)".format(red2, green2, blue2))*/

    for (rgb <- imageRGBVector) {
      val red = (rgb >> 16) & 0xFF
      val green = (rgb >> 8) & 0xFF
      val blue = (rgb) & 0xFF

      // println("(%4d,%4d,%4d)".format(red, green, blue))

      // Calculate the index of this pixel in the histogramVector
      val idx = ((red * (NUM_BINS - 1)) / 256.0).round +
        4 * ((green * (NUM_BINS - 1)) / 256.0).round +
        16 * ((blue * (NUM_BINS - 1)) / 256.0).round

      histogramVector(idx.toInt) += 1
    }

    ROIFeature(histogramVector)
  }

  /**
   * Given a path to an image, extracts feature representation of that image
   */
  def featurizeImage(imgPath: String, regionWidth: Int, regionHeight: Int): DenseMatrix[ROIFeature] = {

    // println("Converting original image to features")

    // Use an additional frame whose thickness is given by this size around the patch
    val PATCH_CONTEXT_SIZE = 0

    val img: BufferedImage = ImageIO.read(new File(imgPath))

    // println("img.size = %d x %d".format(img.getHeight, img.getWidth))

    val xmin = PATCH_CONTEXT_SIZE
    val ymin = PATCH_CONTEXT_SIZE

    val xmax = img.getWidth() - (regionWidth + PATCH_CONTEXT_SIZE)
    val ymax = img.getHeight() - (regionHeight + PATCH_CONTEXT_SIZE)

    val xstep = regionWidth
    val ystep = regionHeight

    // println("xmin, xmax = %d, %d".format(xmin, xmax))
    // println("ymin, ymax = %d, %d".format(ymin, ymax))
    // println("xstep, ystep = %d, %d".format(xstep, ystep))

    val featureMaskWidth = img.getWidth() / xstep
    val featureMaskHeight = img.getHeight() / ystep
    val featureMask = DenseMatrix.zeros[ROIFeature](featureMaskHeight, featureMaskWidth)

    // Upper left of the image is (0, 0)
    for (
      y <- ymin to ymax by ystep;
      x <- xmin to xmax by xstep
    ) {

      // println("Extracting feature at (%d, %d)".format(y, x))

      // Extract a region given by coordinates (x, y) and (x + PATCH_WIDTH, y + PATCH_HEIGHT)
      val patch = img.getSubimage(x, y, regionWidth, regionHeight)

      val xminPc = max(0, x - PATCH_CONTEXT_SIZE)
      val yminPc = max(0, y - PATCH_CONTEXT_SIZE)

      val pcWidth = regionWidth + 2 * PATCH_CONTEXT_SIZE
      val pcHeight = regionHeight + 2 * PATCH_CONTEXT_SIZE

      // In case the patch context window exceeds and/or spills over the boundaries, truncate the window size
      val pcWidthTrunc =
        if (xminPc + pcWidth > img.getWidth())
          img.getWidth() - xminPc
        else
          pcWidth
      val pcHeightTrunc =
        if (yminPc + pcHeight > img.getHeight())
          img.getHeight() - yminPc
        else
          pcHeight

      // Obtain the region, but now with a context
      val patchWithContext = img.getSubimage(xminPc,
        yminPc,
        pcWidthTrunc,
        pcHeightTrunc)

      val patchFeature = histogramFeaturizer(patch, patchWithContext)

      val xf = x / xstep
      val yf = y / ystep
      featureMask(yf, xf) = patchFeature
    }

    // println("Completed - Converting original image to features")

    featureMask
  }

  /**
   * Convert a pixel given in ARGB format to the respective MSRC class label
   */
  def rgbToLabel(rgb: Int): Int = {
    val red = (rgb >> 16) & 0xFF
    val green = (rgb >> 8) & 0xFF
    val blue = (rgb) & 0xFF

    val rgbIndex = blue + (255 * green) + (255 * 255 * red)
    val label = colormap(rgbIndex)

    label
  }

  /**
   * Convert a ground truth image (mask) to a matrix of its classes
   */
  def convertGtImageToLabels(gtImage: BufferedImage): DenseMatrix[Int] = {

    val classMask: DenseMatrix[Int] = DenseMatrix.zeros[Int](gtImage.getHeight(), gtImage.getWidth())
    for (
      x <- 0 until gtImage.getWidth();
      y <- 0 until gtImage.getHeight()
    ) {
      val rgb = gtImage.getRGB(x, y) // Returns in TYPE_INT_ARGB format (4 bytes integer, in ARGB order)
      val label = rgbToLabel(rgb)

      classMask(y, x) = label
    }

    classMask
  }

  /**
   * Given path to the Ground Truth (Image Mask), represent each pixel by its object class
   */
  def featurizeGT(gtPath: String, regionWidth: Int, regionHeight: Int): DenseMatrix[ROILabel] = {

    // println("Converting GT image to features")

    val gtImage: BufferedImage = ImageIO.read(new File(gtPath))

    val xmin = 0
    val ymin = 0

    val xmax = gtImage.getWidth() - regionWidth
    val ymax = gtImage.getHeight() - regionHeight

    val xstep = regionWidth
    val ystep = regionHeight

    val labelMaskWidth = gtImage.getWidth() / regionWidth
    val labelMaskHeight = gtImage.getHeight() / regionHeight
    val labelMask = DenseMatrix.zeros[ROILabel](labelMaskHeight, labelMaskWidth)

    // Upper left of the image is (0, 0)
    for (
      y <- ymin to ymax by ystep;
      x <- xmin to xmax by xstep
    ) {

      // println("Extracting label at (%d, %d, %d, %d)".format(x, y, regionWidth, regionHeight))

      val patch = gtImage.getSubimage(x, y, regionWidth, regionHeight)
      val patchLabelMask = convertGtImageToLabels(patch)

      // Obtain the majority class in this mask
      val majorityLabel = patchLabelMask.toArray.toList
        .groupBy(identity)
        .map { case (k, v) => (k, v.size) }
        .toList
        .sortBy(-_._2)
        .head._1

      val xf = x / xstep
      val yf = y / ystep
      labelMask(yf, xf) = ROILabel(majorityLabel)
    }

    /*
    println(gtPath)
    val gtPathArr = gtPath.split('/')
    val newfname = gtPathArr(gtPathArr.size - 1).split('.')(0)
    val outname = "../debug/%s.jpg".format(newfname)
    printLabeledImage(labelMask, outname)
     */

    // println("Completed - Converting GT image to features")

    labelMask
  }

  /**
   * Returns a LabeledObject instance for an image and its corresponding labeled segments
   */
  def getLabeledObject(imgPath: String, gtPath: String, loadPrecomputedFeatures: Boolean = true): LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]] = {
    if (loadPrecomputedFeatures)
      LabeledObject(roiFileToLabelImage(gtPath), roiFileToFeatureImage(imgPath))
    else
      LabeledObject(featurizeGT(gtPath, REGION_WIDTH, REGION_HEIGHT), featurizeImage(imgPath, REGION_WIDTH, REGION_HEIGHT))
  }

  def loadMSRCDataFromFile(msrcFolder: String, listFileName: String, limit: Int, loadPrecomputedFeatures: Boolean = true): Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]] = {

    val imagesDir: String = if (loadPrecomputedFeatures) msrcFolder + "/ImagesFeatures" else msrcFolder + "/Images"
    val gtDir: String = if (loadPrecomputedFeatures) msrcFolder + "/GroundTruthFeatures" else msrcFolder + "/GroundTruth"

    val data =
      for (imgFilename <- Source.fromURL(getClass.getResource(listFileName)).getLines()) yield {
        val pat = if (loadPrecomputedFeatures) "%s/%s.csv" else "%s/%s"
        val imgPath = pat.format(imagesDir, imgFilename)

        val gtFilename = imgFilename.replace("_s", "_s_GT")
        val gtPath = pat.format(gtDir, gtFilename)

        // println("Training Image = %s\nMask = %s".format(imgPath, gtPath))
        getLabeledObject(imgPath, gtPath, true)
      }

    data.take(limit).toArray
  }

  /**
   * Converts the MSRC dataset into an array of LabeledObjects
   * Requires dataFolder argument should contain two folders: "Images" and "GroundTruth"
   */
  def loadMSRC(msrcFolder: String, trainLimit: Int = 334, testLimit: Int = 256): (Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]], Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]]) = {

    // Split obtained from: http://graphics.stanford.edu/projects/densecrf/unary/
    // (trainSetFilenames uses training and validation sets)
    // These files contains filenames of respective GT images
    // Source.fromURL(getClass.getResource(colormapFile))
    val trainSetFileListPath: String = "/imageseg_train.txt"
    val testSetFileListPath: String = "/imageseg_test.txt"

    val trainData = loadMSRCDataFromFile(msrcFolder, trainSetFileListPath, trainLimit)
    val testData = loadMSRCDataFromFile(msrcFolder, trainSetFileListPath, testLimit)

    (trainData, testData)
  }

  /**
   * Serialize image
   *
   * Format per line:
   * i,j,f_1,f_2,...,f_n
   */
  def roiFeatureImageToFile(inMat: DenseMatrix[ROIFeature], outFile: String): Unit = {
    val writer = new PrintWriter(new File(outFile))

    inMat.keysIterator.foreach {
      case (i, j) =>
        // Convert a vector to a comma-separated list of values in string representation
        val strFeatureVector = inMat(i, j).feature.toArray
          .flatMap { x => "%f".format(x) :: "," :: Nil }
          .dropRight(1)
          .reduceLeft((x, y) => x + y)
        writer.write("%d,%d,%s\n".format(i, j, strFeatureVector))
    }

    writer.close()

  }

  /**
   * Deserialize image from file
   *
   * Format per line:
   * i,j,f_1,f_2,...,f_n
   */
  def roiFileToFeatureImage(inFile: String): DenseMatrix[ROIFeature] = {

    val lines = Source.fromFile(inFile).getLines().toArray

    // Find height and width of image by a full-scan on the lines
    val numRows = lines.map { x => x.split(',')(0).toInt }.max + 1
    val numCols = lines.map { x => x.split(',')(1).toInt }.max + 1

    val img = DenseMatrix.zeros[ROIFeature](numRows, numCols)

    lines.foreach {
      line =>
        val elems = line.split(',')
        val i = elems(0).toInt
        val j = elems(1).toInt
        val vec: DenseVector[Double] = DenseVector(elems.slice(2, elems.size).map(_.toDouble))
        img(i, j) = ROIFeature(vec)
    }

    img
  }

  /**
   * Serialize mask
   *
   * Format per line:
   * i,j,y_{i,j}
   */
  def roiLabelImageToFile(inMat: DenseMatrix[ROILabel], outFile: String): Unit = {
    val writer = new PrintWriter(new File(outFile))

    inMat.keysIterator.foreach {
      case (i, j) =>
        writer.write("%d,%d,%d\n".format(i, j, inMat(i, j).label))
    }

    writer.close()
  }

  /**
   * Deserialize mask
   */
  def roiFileToLabelImage(inFile: String): DenseMatrix[ROILabel] = {

    val lines = Source.fromFile(inFile).getLines().toArray

    // Find height and width of image by a full-scan on the lines
    val numRows = lines.map { x => x.split(',')(0).toInt }.max + 1
    val numCols = lines.map { x => x.split(',')(1).toInt }.max + 1
    // val numClasses = lines.map { x => x.split(',')(2).toInt }.max + 1
    val numClasses = 24

    val img = DenseMatrix.zeros[ROILabel](numRows, numCols)

    lines.foreach {
      line =>
        val elems = line.split(',')
        val i = elems(0).toInt
        val j = elems(1).toInt
        val lab = elems(2).toInt
        img(i, j) = ROILabel(lab, numClasses)
    }

    img

  }

  def featurizeData() = {

    val msrcDir = "../data/generated/MSRC_ObjCategImageDatabase_v2"

    val imagesPath = msrcDir + "/Images"
    val gtPath = msrcDir + "/GroundTruth"
    val imagesOpPath = msrcDir + "/ImagesFeatures"
    val gtOpPath = msrcDir + "/GroundTruthFeatures"

    for (file <- new File(imagesPath).listFiles.toIterator if file.isFile && file.getName().endsWith(".bmp")) {
      println("Processing " + file.getName())
      val imgMat = featurizeImage(file.getAbsolutePath, REGION_WIDTH, REGION_HEIGHT)
      val outFilePath = "%s/%s.csv".format(imagesOpPath, file.getName())
      roiFeatureImageToFile(imgMat, outFilePath)
    }

    for (file <- new File(gtPath).listFiles.toIterator if file.isFile && file.getName().endsWith(".bmp")) {
      println("Processing " + file.getName())
      val gtMat = featurizeGT(file.getAbsolutePath, REGION_WIDTH, REGION_HEIGHT)
      val outFilePath = "%s/%s.csv".format(gtOpPath, file.getName())
      roiLabelImageToFile(gtMat, outFilePath)
    }

  }

  def main(args: Array[String]): Unit = {

    featurizeData()

    /*
    val examples = loadMSRC("../data/generated/MSRC_ObjCategImageDatabase_v2")

    val someExample = examples._1(0)
    val x = someExample.pattern
    val y = someExample.label

    println("x.size = %d x %d".format(x.rows, x.cols))
    println("y.size = %d x %d".format(y.rows, y.cols))
    println("featureSize = %d".format(x(0, 0).feature.size))
    println("numClasses = %d".format(y(0, 0).numClasses))

    val phi = ImageSegmentationDemo.featureFn(x, y)
    println("phi.size = %d".format(phi.size))

    val w = DenseVector.zeros[Double](phi.size)
    val ell = 0
    val ellMat = DenseVector.zeros[Double](phi.size)
    val dummy = new StructSVMModel[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]](w, ell, ellMat,
      ImageSegmentationDemo.featureFn,
      ImageSegmentationDemo.lossFn,
      ImageSegmentationDemo.oracleFn,
      ImageSegmentationDemo.predictFn)
    val dumy = ImageSegmentationDemo.oracleFn(dummy, x, y)
    println("oracle_y.size = %d x %d".format(dumy.rows, dumy.cols))
    * 
    */

  }

}