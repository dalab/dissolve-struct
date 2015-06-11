package ch.ethz.dalab.dissolve.examples.utils

import scala.io.Source
import scala.util.Random

object ExampleUtils {

  /**
   * Obtained adj and noun from: https://github.com/kohsuke/wordnet-random-name
   */
  val adjFilename = "adj.txt"
  val nounFilename = "noun.txt"

  val adjList = Source.fromURL(getClass.getResource("/adj.txt")).getLines().toArray
  val nounList = Source.fromURL(getClass.getResource("/noun.txt")).getLines().toArray

  def getRandomElement[T](lst: Seq[T]): T = lst(Random.nextInt(lst.size))

  def generateExperimentName(prefix: Seq[String] = List.empty, suffix: Seq[String] = List.empty, separator: String = "-"): String = {

    val nameList: Seq[String] = prefix ++ List(getRandomElement(adjList), getRandomElement(nounList)) ++ suffix

    val separatedNameList = nameList
      .flatMap { x => x :: separator :: Nil } // Juxtapose with separators
      .dropRight(1) // Drop the last separator

    separatedNameList.reduce(_ + _)
  }

  def main(args: Array[String]): Unit = {
    println(generateExperimentName())
  }

}