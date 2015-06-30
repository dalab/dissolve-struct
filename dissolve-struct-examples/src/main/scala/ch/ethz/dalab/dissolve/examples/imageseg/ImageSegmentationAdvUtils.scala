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

/**
 * @author torekond
 */
object ImageSegmentationAdvUtils {

  /**
   * === Paths ===
   */

  val colormapFile = "/imageseg_label_color_map.txt"

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

    // Get size of unary features
    val d = lineToValues(Source.fromFile(unaryFile.toFile())
      .getLines
      .next())
      .size

    val unaryMatrix = DenseMatrix.zeros[Double](d, n)

    // Second pass, get the super-pixels
    for ((line, idx) <- Source.fromFile(unaryFile.toFile()).getLines.zipWithIndex) {
      val featureVector = DenseVector(lineToValues(line))
      unaryMatrix(::, idx) := featureVector
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
   */
  def loadData(dataDir: String = "../data/generated/msrc",
               split: String = "Train",
               limit: Int = Integer.MAX_VALUE): Array[LabeledObject[QuantizedImage, QuantizedLabel]] = {

    val splitFilename: String = "%s.txt".format(split)
    val splitFilePath: Path = Paths.get(dataDir, splitFilename)

    val labelsDir: Path = Paths.get(dataDir, "labels")
    val splitDir: Path = Paths.get(dataDir, split)

    val labeledObjectSeq =
      for (imgFilenameRaw <- Source.fromFile(splitFilePath.toFile()).getLines.take(limit)) yield {
        val imgName = imgFilenameRaw.trim().stripLineEnd.replace(".bmp", "")
        println(imgName)

        // Get Width and Height
        val imgFilename = imgName + ".bmp"
        val imgPath = splitDir.resolve(imgFilename)
        val img = ImageIO.read(imgPath.toFile())
        val width = img.getWidth()
        val height = img.getHeight()

        // Read Unaries
        val unaryName = imgName + ".local1"
        val unaryFile = splitDir.resolve(unaryName)
        val unaries: DenseMatrix[Double] = deserializeUnaryTerm(unaryFile)

        // Read Pairwise
        val pairwiseName = imgName + ".edges"
        val pairwiseFile = splitDir.resolve(pairwiseName)
        val pairwise: AdjacencyList = deserializePairwiseTerm(pairwiseFile)

        // Read Mapping
        val mappingName = imgName + ".map"
        val mappingFile = splitDir.resolve(mappingName)
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

    labeledObjectSeq.toArray
  }

  def labelsToImage(labelArray: Array[Label], width: Int, height: Int): BufferedImage = {
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    img.setRGB(0, 0, width, height, labelArray.map(i => labelToRGB(i)), 0, width)

    img
  }

  /**
   * Output super-pixel labels as an image
   *
   * z(i) = j : Super-pixel idx = `i` and pixel idx = `j`
   */
  def printQuantizedLabel(y: QuantizedLabel,
                          z: Array[Int],
                          width: Int,
                          height: Int): Unit = {

    val debugDir = Paths.get("../data/generated/msrc", "debug")
    val ext = "bmp"
    if (!debugDir.toFile().exists())
      debugDir.toFile().mkdir()

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

    val pixelLabelImage = labelsToImage(pixelLabels, width, height)
    val imageOutName = y.filename + "." + ext
    val imageOutFile = debugDir.resolve(imageOutName).toFile()

    ImageIO.write(pixelLabelImage, ext, imageOutFile)

  }

  def main(args: Array[String]): Unit = {

    println("Loading data")
    val data = loadData(limit = 1)

    println("Writing images")
    data.foreach {
      case lo =>
        val x = lo.pattern
        val y = lo.label

        println(x.filename)
        printQuantizedLabel(y, x.pixelMapping, x.width, x.height)

    }

  }

}