[![Build Status](https://travis-ci.org/dalab/dissolve-struct.svg?branch=tuning)](https://travis-ci.org/dalab/dissolve-struct)
[![Release status](https://img.shields.io/badge/release-v0.1-orange.svg)](https://github.com/dalab/dissolve-struct/releases)

dissolve<sup>struct</sup>
===========

Distributed solver library for structured output prediction, based on Spark.

The library is based on the primal-dual BCFW solver, allowing approximate inference oracles, and distributes this algorithm using the recent communication efficient CoCoA scheme.
The interface to the user is the same as in the widely used SVM<sup>struct</sup> in the single machine case.

For more information, checkout the [project page](http://dalab.github.io/dissolve-struct/)

