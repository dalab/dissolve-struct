package ch.ethz.dalab.dissolve.examples.imageseg

/**
 * @author torekond
 */
object ImageSegTypes {
  type Label = Int
  type Index = Int
  type SuperIndex = Int
  type RGB_INT = Int
  type AdjacencyList = Array[Array[SuperIndex]]
}