---
layout: article
title: "Binary SVM"
modified:
categories: examples
excerpt:
tags: []
image:
  feature:
  teaser: bsvm_thumb.png
  thumb: bsvm_thumb.png
date: 2015-05-24T22:57:28+02:00
---

{% include toc.html %}

## Datasets
In the `ch.ethz.dalab.dissolve.examples.binaryclassification` package of the
`dissolve-struct` package, you'll find three Binary SVM examples using 3 datasets:

1. Adult ([**A1A**](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a1a))
2. Forest Cover ([**COV**](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#covtype.binary))
3. Reuters Corpus Volume 1 ([**RCV1**](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#rcv1.binary))

Each of these intend to display different aspects of dissolve<sup>struct</sup>'s
awesomeness.
**COV** is a relatively large corpus containing around 581,012 data points, each with
54 features.
**RCV1** contains 20,242 data points, but with each example involving a sparse vector
with 47,236 features.


## Running the examples
Training a binary SVM locally from the command-line is done as follows, here for the Forest Cover (COV) dataset. Within `dissolve-struct-examples` directory, run
{% highlight bash %}
spark-1.X/bin/spark-submit \
	--class "ch.ethz.dalab.dissolve.examples.binaryclassification.COVBinary" \
	--master local \
	--driver-memory 2G \
	<examples-jar-path>
{% endhighlight %}


## Running your own Binary classifier
A Binary classifier is bundled with dissolve<sup>struct</sup>.
To use it, you'll merely need to provide the data and the solver parameters.
Just like any other Spark MLLib classifiers, the data can be provided
using the `loadLibSVMFile` format.

{% highlight scala %}
val training = MLUtils.loadLibSVMFile(sc, covPath)
val solverOptions: SolverOptions[Vector[Double], Double] = new SolverOptions()

val model = BinarySVMWithDBCFW.train(training, solverOptions)
{% endhighlight %}

**Label Format:** The labels need to be +1.0/-1.0. This can be usually taken care
of in the preprocessing stage.
{: .notice-warning}
