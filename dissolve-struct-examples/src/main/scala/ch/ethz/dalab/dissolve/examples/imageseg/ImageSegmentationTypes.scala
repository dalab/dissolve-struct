package ch.ethz.dalab.dissolve.examples.imageseg

/**
 * @author torekond
 */
object ImageSegmentationTypes {
  type Label = Int
  type Index = Int
  type SuperIndex = Int
  type RGB_INT = Int
  type AdjacencyList = Array[Array[SuperIndex]]
}