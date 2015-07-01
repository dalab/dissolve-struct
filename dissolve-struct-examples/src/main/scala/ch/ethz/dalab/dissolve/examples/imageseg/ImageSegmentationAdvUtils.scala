package ch.ethz.dalab.dissolve.examples.imageseg

import java.awt.image.BufferedImage
import java.nio.file.Path
import java.nio.file.Paths
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import breeze.linalg.DenseMatrix
import breeze.linalg.DenseVector
import ch.ethz.dalab.dissolve.examples.imageseg.ImageSegmentationTypes._
import ch.ethz.dalab.dissolve.regression.LabeledObject
import javax.imageio.ImageIO
import java.io.File
import breeze.linalg.max
import java.awt.Graphics2D
import java.awt.Font
import java.awt.Color

/**
 * @author torekond
 */
object ImageSegmentationAdvUtils {

  /**
   * === Paths ===
   */

  val colormapFile = "/imageseg_label_color_map.txt"
  val legendImage = "/msrc_legend.bmp"

  /**
   * Map from a label to the corresponding RGB pixel
   */
  val labelToRGB: Map[Label, RGB_INT] = Source.fromURL(getClass.getResource(colormapFile))
    .getLines()
    .map { line => line.split(" ") }
    .map {
      case Array(label, value, r, g, b, className, labelIdx) =>
        val rgb = ((r.toInt & 0x0ff) << 16) |
          ((g.toInt & 0x0ff) << 8) |
          (b.toInt & 0x0ff)
        labelIdx.toInt -> rgb
    }
    .toMap

  /**
   * Deserialize Unary (Local Data evidence) terms
   */
  def deserializeUnaryTerm(unaryFile: Path): DenseMatrix[Double] = {

    def lineToValues(line: String): Array[Double] = line.split(" ")
      .filter { x => x.length() > 0 }
      .map(_.toDouble)

    // First pass, get number of super-pixels
    val n: Int = Source.fromFile(unaryFile.toFile()).getLines.size

    // Get size of unary features (In this case. scores of each class
    // except background)
    val d = lineToValues(Source.fromFile(unaryFile.toFile())
      .getLines
      .next())
      .size

    val unaryMatrix = DenseMatrix.zeros[Double](d + 1, n)

    // Second pass, get the super-pixels
    for ((line, idx) <- Source.fromFile(unaryFile.toFile()).getLines.zipWithIndex) {
      val featureVector = DenseVector(lineToValues(line))
      unaryMatrix(0 until d, idx) := featureVector
      unaryMatrix(d, idx) = max(featureVector)
    }

    unaryMatrix
  }

  /**
   * Deserialize Pairwise (Spatial) terms
   */
  def deserializePairwiseTerm(pairwiseFile: Path): AdjacencyList = {

    def lineToIndices(line: String): Array[SuperIndex] = {
      val superIdx = line.split(" ")(0)
      if (line.split(" ").size == 1)
        Array() // This node has no edges
      else
        line.split(" ")(1)
          .split(",")
          .map(_.toInt)
    }

    val arb = new ArrayBuffer[Array[SuperIndex]]

    for (line <- Source.fromFile(pairwiseFile.toFile()).getLines)
      arb += lineToIndices(line)

    arb.toArray
  }

  /**
   * Deserialize mapping of super-pixel index to pixel index
   */
  def deserializeMapping(mappingFile: Path,
                         width: Int,
                         height: Int): Array[Index] = {
    // First pass - Calculate the number of pixels
    val n = Source.fromFile(mappingFile.toFile())
      .getLines
      .map(_.split(","))
      .map { x => x.map(_.toInt).max } // Calculate max in each row
      .max + 1

    val pixelMapping: Array[Index] = Array.fill(n)(0)

    // Second pass - create pixel to super-pixel mapping
    Source.fromFile(mappingFile.toFile())
      .getLines
      .map(_.split(","))
      .map { x => x.map(_.toInt) }
      .foreach {
        case line =>
          val superIdx = line(0)
          line.slice(1, line.length).foreach {
            case idx =>
              pixelMapping(idx) = superIdx
          }
      }

    // Convert this pixel-mapping to column-major
    val pixelMappingColMajor = DenseMatrix(pixelMapping).reshape(height, width).t.toArray

    pixelMappingColMajor
  }

  /**
   * Deserialize labels
   */
  def deserializeLabels(labelsFile: Path): Array[Label] =
    Source.fromFile(labelsFile.toFile())
      .getLines
      .map(_.trim().stripLineEnd)
      .map(_.toInt)
      .toArray

  /**
   * Load Data from disk
   *
   * Loads filenames specified in `split` from the directory `All`
   */
  def loadData(dataDir: String = "../data/generated/msrc",
               splitFilePath: Path,
               limit: Int = Integer.MAX_VALUE): Array[LabeledObject[QuantizedImage, QuantizedLabel]] = {

    assert(splitFilePath.toFile().exists(), "splitFilePath.toFile().exists()")

    val labelsDir: Path = Paths.get(dataDir, "labels")
    val filesDir: Path = Paths.get(dataDir, "All")

    val labeledObjectSeq =
      for (imgFilenameRaw <- Source.fromFile(splitFilePath.toFile()).getLines) yield {
        val imgName = imgFilenameRaw.trim().stripLineEnd.replace(".bmp", "")
        println(imgName)

        // Get Width and Height
        val imgFilename = imgName + ".bmp"
        val imgPath = filesDir.resolve(imgFilename)
        val img = ImageIO.read(imgPath.toFile())
        val width = img.getWidth()
        val height = img.getHeight()

        // Read Unaries
        val unaryName = imgName + ".local1"
        val unaryFile = filesDir.resolve(unaryName)
        val unaries: DenseMatrix[Double] = deserializeUnaryTerm(unaryFile)

        // Read Pairwise
        val pairwiseName = imgName + ".edges"
        val pairwiseFile = filesDir.resolve(pairwiseName)
        val pairwise: AdjacencyList = deserializePairwiseTerm(pairwiseFile)

        // Read Mapping
        val mappingName = imgName + ".map"
        val mappingFile = filesDir.resolve(mappingName)
        val pixelMapping: Array[Index] = deserializeMapping(mappingFile, width, height)

        // Read Labels
        val labelsName = imgName + ".txt"
        val labelsFile = labelsDir.resolve(labelsName)
        val labels: Array[Label] = deserializeLabels(labelsFile)

        val x = QuantizedImage(unaries, pairwise, pixelMapping, width, height, imgName)
        val y = QuantizedLabel(labels, imgName)
        val lo = LabeledObject(y, x)

        lo
      }

    labeledObjectSeq
      .filter {
        case lo =>
          val labels = lo.label.labels
          // 22 - Mountain
          // 23 - Horse
          !(labels.contains(22) || labels.contains(23))
      }
      .take(limit)
      .toArray
  }

  def labelsToImage(labelArray: Array[Label], width: Int, height: Int): BufferedImage = {
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    img.setRGB(0, 0, width, height, labelArray.map(i => labelToRGB(i)), 0, width)

    img
  }

  def getImageWithText(w: Int, h: Int, text: String): BufferedImage = {
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val g2: Graphics2D = img.createGraphics()
    // Background
    g2.setPaint(Color.WHITE)
    g2.fillRect(0, 0, w, h)

    g2.setPaint(Color.black)
    g2.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 10))
    val ymargin = 15
    val offset = 15

    for ((str, idx) <- text.split('\n').zipWithIndex) {
      g2.drawString(str, 10, ymargin + offset * idx)
    }

    g2.dispose()

    img
  }

  /**
   * Places images into a 2 x 2 layout
   * Assume all images are of same size
   */
  def printImageTile(x11: BufferedImage,
                     x12: BufferedImage,
                     x21: BufferedImage,
                     x22: BufferedImage): BufferedImage = {

    val legend = new File("../data/generated/msrc/msrc_legend.bmp")
    val legendImage = ImageIO.read(legend)
    val w_leg = legendImage.getWidth()
    val h_leg = legendImage.getHeight()

    val offset: Int = 10
    val margin: Int = 10
    val w_x: Int = x11.getWidth()
    val h_x: Int = x11.getHeight()

    val w = max(2 * w_x, w_leg) + offset + 2 * margin
    val h = 2 * h_x + h_leg + 2 * offset + 2 * margin

    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val g2: Graphics2D = img.createGraphics()
    val oldCol = g2.getColor()

    g2.setPaint(Color.WHITE)
    g2.fillRect(0, 0, w, h)
    g2.setColor(oldCol)

    g2.drawImage(x11, null, margin, margin)
    g2.drawImage(x12, null, w_x + offset + margin, margin)
    g2.drawImage(x21, null, margin, h_x + offset + margin)
    g2.drawImage(x22, null, w_x + offset + margin, h_x + offset + margin)

    val centeredx = w / 2 - w_leg / 2
    g2.drawImage(legendImage, null, centeredx, 2 * (h_x + offset) + margin)

    g2.dispose()

    img
  }

  /**
   * Output super-pixel labels as an image
   *
   * z(i) = j : Super-pixel idx = `i` and pixel idx = `j`
   */
  def getQuantizedLabelImage(y: QuantizedLabel,
                             z: Array[Int],
                             width: Int,
                             height: Int): BufferedImage = {

    val n = y.labels.size // # Super pixels
    val N = z.size // # Pixels

    val pixelLabels = Array.fill(N)(0)

    for (i <- 0 until N) {
      val pixelIdx = i
      val superPixelIdx = z(pixelIdx)
      val superPixelLabel = y.labels(superPixelIdx)
      pixelLabels(pixelIdx) = superPixelLabel
    }

    val foo = pixelLabels.groupBy(identity).map(x => (x._1, x._2.size))
    println(foo)

    println("w = %d, h = %d".format(width, height))

    labelsToImage(pixelLabels, width, height)
  }

  def writeImage(img: BufferedImage, outFilePath: String): Unit = {
    ImageIO.write(img, "bmp", new File(outFilePath))
  }

  def main(args: Array[String]): Unit = {

    println("Loading data")

    val dataDir: String = "../data/generated/msrc"
    val trainFilePath: Path = Paths.get(dataDir, "Train.txt")
    val data = loadData(dataDir, trainFilePath, limit = 1)

    val debugDir = Paths.get(dataDir, "debug")
    if (!debugDir.toFile().exists())
      debugDir.toFile().mkdir()

    println("Writing images")
    data.foreach {
      case lo =>
        val x = lo.pattern
        val y = lo.label

        println(x.filename)
        val imageOutName = y.filename + ".bmp"
        val imageOutFile = debugDir.resolve(imageOutName)
        writeImage(getQuantizedLabelImage(y, x.pixelMapping, x.width, x.height),
          imageOutFile.toString())

    }

  }

}