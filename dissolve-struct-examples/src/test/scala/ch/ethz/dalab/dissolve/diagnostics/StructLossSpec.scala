package ch.ethz.dalab.dissolve.diagnostics

/**
 * @author torekond
 */
class StructLossSpec extends UnitSpec {

  val NUM_ATTEMPTS = 100 // = # times each test case is attempted

  // A sample datapoint
  val lo = data(0)
  // Size of joint feature map
  val d = phi(lo.pattern, lo.label).size
  // No. of data examples
  val M = data.length

  "Δ(y, y)" should "= 0" in {

    val lossSeq: Seq[Double] = for (k <- 0 until NUM_ATTEMPTS) yield {

      // Choose a random example
      val m = scala.util.Random.nextInt(M)
      val lo = data(m)
      val x_m = lo.pattern
      val y_m = lo.label

      delta(y_m, y_m)

    }

    // This should be empty
    val uneqLossSeq: Seq[Double] = lossSeq.filter(_ != 0.0)

    assert(uneqLossSeq.length == 0,
      "%d / %d cases failed".format(uneqLossSeq.length, lossSeq.length))
  }

  "Δ(y, y')" should ">= 0" in {

    val lossSeq: Seq[Double] = for (k <- 0 until NUM_ATTEMPTS) yield {

      // Choose a random example
      val m = scala.util.Random.nextInt(M)
      val lo = data(m)
      val x_m = lo.pattern
      val degree = scala.util.Random.nextDouble()
      val y_m = perturb(lo.label, degree)

      delta(y_m, y_m)

    }

    // This should be empty
    val uneqLossSeq: Seq[Double] = lossSeq.filter(_ < 0.0)

    assert(uneqLossSeq.length == 0,
      "%d / %d cases failed".format(uneqLossSeq.length, lossSeq.length))
  }

}