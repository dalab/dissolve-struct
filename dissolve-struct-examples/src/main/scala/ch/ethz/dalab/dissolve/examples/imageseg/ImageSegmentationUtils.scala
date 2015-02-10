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

object ImageSegmentationUtils {

  val featurizer_options: List[String] = List("HIST")

  val colormapFile = "imageseg_colormap.txt"
  val colormap: Map[Int, Int] = Source.fromURL(getClass.getResource(colormapFile))
    .getLines()
    .map { line => line.split(" ") }
    .map {
      case Array(label, value, r, g, b, className) =>
        value.toInt -> label.toInt
    }
    .toMap

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
    val imageRGBVector: Array[Int] = patchWithContext.getRaster()
      .getDataBuffer().asInstanceOf[DataBufferInt]
      .getData()

    for (rgb <- imageRGBVector) {
      val red = (rgb >> 16) & 0xFF
      val green = (rgb >> 8) & 0xFF
      val blue = (rgb) & 0xFF

      // Calculate the index of this pixel in the histogramVector
      val idx = ((red * NUM_BINS) / 256) +
        4 * ((green * NUM_BINS) / 256) +
        16 * ((blue * NUM_BINS) / 256)
      histogramVector(idx) += 1
    }

    ROIFeature(histogramVector)
  }

  /**
   * Given a path to an image, extracts feature representation of that image
   */
  def featurizeImage(imgPath: String, regionWidth: Int, regionHeight: Int): Matrix[ROIFeature] = {

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
    val featureMaskHeight = img.getHeight() / xstep
    val featureMask = DenseMatrix.zeros[ROIFeature](ymax, xmax)

    // Upper left of the image is (0, 0)
    for (
      y <- ymin until ymax by ystep;
      x <- xmin until xmax by xstep
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
  def featurizeGT(gtPath: String, regionWidth: Int, regionHeight: Int): Matrix[ROILabel] = {

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
      y <- ymin until ymax by ystep;
      x <- xmin until xmax by xstep
    ) {

      val patch = gtImage.getSubimage(x, y, regionWidth, regionHeight)
      val patchLabelMask = convertGtImageToLabels(gtImage)

      // Obtain the majority class in this mask
      val majorityLabel = patchLabelMask.toArray.toList
        .groupBy(identity)
        .map { case (k, v) => (k, v.size) }
        .toList
        .sortBy(-_._2)
        .head._1

      val xf = x / xstep
      val yf = y / ystep
      labelMask(y, x) = ROILabel(majorityLabel)
    }

    labelMask
  }

  /**
   * Returns a LabeledObject instance for an image and its corresponding labeled segments
   */
  def getLabeledObject(imgPath: String, gtPath: String): LabeledObject[Matrix[ROIFeature], Matrix[ROILabel]] = {

    val REGION_WIDTH = 5
    val REGION_HEIGHT = 5

    LabeledObject(featurizeGT(gtPath, REGION_WIDTH, REGION_HEIGHT), featurizeImage(imgPath, REGION_WIDTH, REGION_HEIGHT))
  }

  /**
   * Converts the MSRC dataset into an array of LabeledObjects
   * Requires dataFolder argument should contain two folders: "Images" and "GroundTruth"
   */
  def loadMSRC(msrcFolder: String): Array[LabeledObject[Matrix[ROIFeature], Matrix[ROILabel]]] = {

    // Split obtained from: http://graphics.stanford.edu/projects/densecrf/unary/
    // (trainSetFilenames uses training and validation sets)
    // These files contains filenames of respective GT images
    val trainSetFileListPath: String = "../data/imageseg_train.txt"
    val testSetFileListPath: String = "../data/imageseg_test.txt"

    val imagesDir: String = msrcFolder + "/Images"
    val gtDir: String = msrcFolder + "/GroundTruth"

    val data =
      for (imgFilename <- Source.fromFile(trainSetFileListPath).getLines().slice(0, 1)) yield {
        val imgPath = "%s/%s".format(imagesDir, imgFilename)

        val gtFilename = imgFilename.replace("_s", "_s_GT")
        val gtPath = "%s/%s".format(gtDir, gtFilename)

        getLabeledObject(imgPath, gtPath)
      }

    data.toArray
  }

  def main(args: Array[String]): Unit = {
    loadMSRC("../data/generated/MSRC_ObjCategImageDatabase_v2")
  }

}