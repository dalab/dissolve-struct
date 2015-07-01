package ch.ethz.dalab.dissolve.examples.imageseg

import java.nio.file.Paths
import org.apache.log4j.PropertyConfigurator
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import ch.ethz.dalab.dissolve.classification.StructSVMModel
import ch.ethz.dalab.dissolve.classification.StructSVMWithDBCFW
import ch.ethz.dalab.dissolve.utils.cli.CLAParser
import javax.imageio.ImageIO

/**
 * @author torekond
 */
object ImageSegmentationAdvRunner {

  def main(args: Array[String]): Unit = {

    PropertyConfigurator.configure("conf/log4j.properties")
    System.setProperty("spark.akka.frameSize", "512")

    /**
     * Load all options
     */
    val (solverOptions, kwargs) = CLAParser.argsToOptions[QuantizedImage, QuantizedLabel](args)
    val dataDir = kwargs.getOrElse("input_path", "../data/generated/msrc")
    val appname = kwargs.getOrElse("appname", "imageseg-%d".format(System.currentTimeMillis() / 1000))
    val debugPath = kwargs.getOrElse("debug_file", "imageseg-%d.csv".format(System.currentTimeMillis() / 1000))
    solverOptions.debugInfoPath = debugPath

    println(dataDir)
    println(kwargs)

    solverOptions.doLineSearch = true

    /**
     * Setup Spark
     */
    val conf = new SparkConf().setAppName(appname).setMaster("local")
    val sc = new SparkContext(conf)
    sc.setCheckpointDir("checkpoint-files")

    val trainDataSeq = ImageSegmentationAdvUtils.loadData(dataDir, "Train", 50)
    val testDataSeq = ImageSegmentationAdvUtils.loadData(dataDir, "Test", 5)

    val trainData = sc.parallelize(trainDataSeq, 1).cache
    val testData = sc.parallelize(testDataSeq, 1).cache

    solverOptions.testDataRDD = Some(testData)

    println(solverOptions)

    val trainer: StructSVMWithDBCFW[QuantizedImage, QuantizedLabel] =
      new StructSVMWithDBCFW[QuantizedImage, QuantizedLabel](
        trainData,
        ImageSegmentationAdv,
        solverOptions)

    val model: StructSVMModel[QuantizedImage, QuantizedLabel] = trainer.trainModel()

    // Create directories for image out, if it doesn't exist
    val imageOutDir = Paths.get(dataDir, "debug", appname)
    if (!imageOutDir.toFile().exists())
      imageOutDir.toFile().mkdir()

    println("Test time!")
    for (lo <- trainDataSeq) {
      val t0 = System.currentTimeMillis()
      val prediction = model.predict(lo.pattern)
      val t1 = System.currentTimeMillis()

      val filename = lo.pattern.filename
      val format = "bmp"
      val outPath = Paths.get(imageOutDir.toString(), "train-%s.%s".format(filename, format))

      // Image
      val imgPath = Paths.get(dataDir.toString(), "Train", "%s.bmp".format(filename))
      val img = ImageIO.read(imgPath.toFile())

      val width = img.getWidth()
      val height = img.getHeight()

      // Write loss info
      val predictTime: Long = t1 - t0
      val loss: Double = ImageSegmentationAdv.lossFn(prediction, lo.label)
      val text = "filename = %s\nprediction time = %d ms\nerror = %f\n#spx = %d".format(filename, predictTime, loss, lo.pattern.unaries.cols)
      val textInfoImg = ImageSegmentationAdvUtils.getImageWithText(width, height, text)

      // GT
      val gtImage = ImageSegmentationAdvUtils.getQuantizedLabelImage(lo.label,
        lo.pattern.pixelMapping,
        lo.pattern.width,
        lo.pattern.height)

      // Prediction
      val predImage = ImageSegmentationAdvUtils.getQuantizedLabelImage(prediction,
        lo.pattern.pixelMapping,
        lo.pattern.width,
        lo.pattern.height)

      val prettyOut = ImageSegmentationAdvUtils.printImageTile(img, gtImage, textInfoImg, predImage)
      ImageSegmentationAdvUtils.writeImage(prettyOut, outPath.toString())

    }

    /*
    for (lo <- testDataSeq) {
      val t0 = System.currentTimeMillis()
      val prediction = model.predict(lo.pattern)
      val t1 = System.currentTimeMillis()

      val filename = lo.pattern.filename
      val format = "bmp"
      val debugFilename = imageOutDir.resolve("test-%s.%s".format(filename, format))

      ImageSegmentationAdvUtils.printQuantizedLabel(lo.label,
        lo.pattern.pixelMapping,
        lo.pattern.width,
        lo.pattern.height)

    }*/

  }

}