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
import breeze.linalg.normalize
import breeze.math._
import breeze.numerics._
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

  val labFreqFile =
    if (ImageSegmentationDemo.RUN_SYNTH)
      "/imageseg_synth_lab_freq.txt"
    else
      "/imageseg_cattle_lab_freq.txt"

  val labFreqMap: Map[Int, Double] = Source.fromURL(getClass.getResource(labFreqFile))
    .getLines()
    .map { line => line.split(',') }
    .map {
      case Array(lab, labFreq) =>
        lab.toInt -> labFreq.toDouble
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

    ImageIO.write(out, "bmp", new File(outputFile))

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
    val NUM_BINS = 8
    // Store the histogram in 3 blocks of [ R | G | B ]
    // The index in the histogram feature vector is function of R,G,B intensity bins
    val histogramVector = DenseVector.zeros[Double](NUM_BINS * NUM_BINS * NUM_BINS)

    val imageRGBVector = patch.getRGB(0, 0, patch.getWidth, patch.getHeight, null, 0, patch.getWidth)

    for (rgb <- imageRGBVector) {
      val red = (rgb >> 16) & 0xFF
      val green = (rgb >> 8) & 0xFF
      val blue = (rgb) & 0xFF

      // Calculate the index of this pixel in the histogramVector
      val idx = math.pow(NUM_BINS, 0) * ((red * (NUM_BINS - 1)) / 256.0).round +
        math.pow(NUM_BINS, 1) * ((green * (NUM_BINS - 1)) / 256.0).round +
        math.pow(NUM_BINS, 2) * ((blue * (NUM_BINS - 1)) / 256.0).round

      histogramVector(idx.toInt) += 1
    }

    ROIFeature(normalize(histogramVector))
  }

  /**
   * Given a path to an image, extracts feature representation of that image
   */
  def featurizeImage(imgPath: String, regionWidth: Int, regionHeight: Int): DenseMatrix[ROIFeature] = {

    // Use an additional frame whose thickness is given by this size around the patch
    val PATCH_CONTEXT_SIZE = 0

    val img: BufferedImage = ImageIO.read(new File(imgPath))

    val xmin = PATCH_CONTEXT_SIZE
    val ymin = PATCH_CONTEXT_SIZE

    val xmax = img.getWidth() - (regionWidth + PATCH_CONTEXT_SIZE)
    val ymax = img.getHeight() - (regionHeight + PATCH_CONTEXT_SIZE)

    val xstep = regionWidth
    val ystep = regionHeight

    val featureMaskWidth = img.getWidth() / xstep
    val featureMaskHeight = img.getHeight() / ystep
    val featureMask = DenseMatrix.zeros[ROIFeature](featureMaskHeight, featureMaskWidth)

    // Upper left of the image is (0, 0)
    for (
      y <- ymin to ymax by ystep;
      x <- xmin to xmax by xstep
    ) {

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

    val suffix = if (ImageSegmentationDemo.RUN_SYNTH) "Synth" else ""

    val imagesDir: String = { if (loadPrecomputedFeatures) msrcFolder + "/ImagesFeatures" else msrcFolder + "/Images" } + suffix
    val gtDir: String = { if (loadPrecomputedFeatures) msrcFolder + "/GroundTruthFeatures" else msrcFolder + "/GroundTruth" } + suffix

    val data =
      for (imgFilename <- Source.fromURL(getClass.getResource(listFileName)).getLines()) yield {
        val pat = if (loadPrecomputedFeatures) "%s/%s.csv" else "%s/%s"
        val imgPath = pat.format(imagesDir, imgFilename)

        val gtFilename = imgFilename.replace("_s", "_s_GT")
        val gtPath = pat.format(gtDir, gtFilename)

        getLabeledObject(imgPath, gtPath, true)
      }

    data.take(limit).toArray
  }

  /**
   * Converts the MSRC dataset into an array of LabeledObjects
   * Requires dataFolder argument should contain two folders: "Images" and "GroundTruth"
   */
  def loadMSRC(msrcFolder: String, trainLimit: Int = 334, testLimit: Int = 256): (Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]], Array[LabeledObject[DenseMatrix[ROIFeature], DenseMatrix[ROILabel]]]) = {

    val suffix = if (ImageSegmentationDemo.RUN_SYNTH) "_synth" else "_cattle"

    // Split obtained from: http://graphics.stanford.edu/projects/densecrf/unary/
    // (trainSetFilenames uses training and validation sets)
    // These files contains filenames of respective GT images
    // Source.fromURL(getClass.getResource(colormapFile))
    // val trainSetFileListPath: String = "/imageseg_train.txt"
    // val testSetFileListPath: String = "/imageseg_test.txt"
    val trainSetFileListPath: String = "/imageseg%s_train.txt".format(suffix)
    val testSetFileListPath: String = "/imageseg%s_test.txt".format(suffix)

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

    def pathToFileName(path: String): String = path.split('/').last

    lines.foreach {
      line =>
        val elems = line.split(',')
        val i = elems(0).toInt
        val j = elems(1).toInt
        val vec: DenseVector[Double] = DenseVector(elems.slice(2, elems.size).map(_.toDouble))
        img(i, j) = ROIFeature(vec, name = pathToFileName(inFile))
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
        img(i, j) = ROILabel(lab, numClasses, labFreqMap(lab))
    }

    img

  }

  def featurizeData() = {

    val msrcDir = "../data/generated/MSRC_ObjCategImageDatabase_v2"

    val suffix = if (ImageSegmentationDemo.RUN_SYNTH) "Synth" else ""

    val imagesPath = msrcDir + "/Images" + suffix
    val gtPath = msrcDir + "/GroundTruth" + suffix
    val imagesOpPath = msrcDir + "/ImagesFeatures" + suffix
    val gtOpPath = msrcDir + "/GroundTruthFeatures" + suffix

    for (
      file <- new File(imagesPath).listFiles.toIterator if file.isFile
        && file.getName().endsWith(".bmp")
    ) {
      println("Processing " + file.getName())
      val imgMat = featurizeImage(file.getAbsolutePath, REGION_WIDTH, REGION_HEIGHT)
      val outFilePath = "%s/%s.csv".format(imagesOpPath, file.getName())
      roiFeatureImageToFile(imgMat, outFilePath)
    }

    for (
      file <- new File(gtPath).listFiles.toIterator if file.isFile
        && file.getName().endsWith(".bmp")
    ) {
      println("Processing " + file.getName())
      val gtMat = featurizeGT(file.getAbsolutePath, REGION_WIDTH, REGION_HEIGHT)
      val outFilePath = "%s/%s.csv".format(gtOpPath, file.getName())
      roiLabelImageToFile(gtMat, outFilePath)
    }

  }

  def main(args: Array[String]): Unit = {

    featurizeData()

  }

}