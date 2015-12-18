package ch.ethz.dalab.dissolve.diagnostics

/**
 * @author torekond
 */
class FeatureFnSpec extends UnitSpec {

  val NUM_ATTEMPTS = 100 // = # times each test case is attempted

  // A sample datapoint
  val lo = data(0)
  // Size of joint feature map
  val d = phi(lo.pattern, lo.label).size
  // No. of data examples
  val M = data.length

  "dim( ϕ(x_m, y_m) )" should "be fixed for all GIVEN (x_m, y_m)" in {

    val dimDiffSeq: Seq[Int] = for (k <- 0 until NUM_ATTEMPTS) yield {

      // Choose a random example
      val m = scala.util.Random.nextInt(M)
      val lo = data(m)
      val x_m = lo.pattern
      val y_m = lo.label

      phi(x_m, y_m).size - d

    }

    // This should be empty
    val uneqDimDiffSeq: Seq[Int] = dimDiffSeq.filter(_ != 0)

    assert(uneqDimDiffSeq.length == 0,
      "%d / %d cases failed".format(uneqDimDiffSeq.length, dimDiffSeq.length))

  }

  it should "be fixed for all PERTURBED (x_m, y_m)" in {

    val dimDiffSeq: Seq[Int] = for (k <- 0 until NUM_ATTEMPTS) yield {

      // Choose a random example
      val m = scala.util.Random.nextInt(M)
      val lo = data(m)
      val x_m = lo.pattern
      val y_m = perturb(lo.label, 0.5)

      phi(x_m, y_m).size - d

    }

    // This should be empty
    val uneqDimDiffSeq: Seq[Int] = dimDiffSeq.filter(_ != 0)

    assert(uneqDimDiffSeq.length == 0,
      "%d / %d cases failed".format(uneqDimDiffSeq.length, dimDiffSeq.length))

  }
}