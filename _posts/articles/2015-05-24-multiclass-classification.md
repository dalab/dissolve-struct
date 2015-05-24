---
layout: article
title: "Multiclass Classification"
modified:
categories: examples
excerpt:
tags: []
image:
  feature: multiclass.png
  teaser: multiclass.png
  thumb: multiclass.png
date: 2015-05-24T23:53:27+02:00
---


{% include toc.html %}


## Running the example
The provided multiclass example is in package `ch.ethz.dalab.dissolve.examples.multiclass`.
This is based on the [Forest Cover multiclass dataset](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#covtype).

{% highlight bash %}
spark-1.X/bin/spark-submit \
	--class "ch.ethz.dalab.dissolve.examples.multiclass.COVMulticlass" \
	--master local \
	--driver-memory 2G \
	<examples-jar-path>
{% endhighlight %}


## Custom Multiclass classifier
For working with a custom K-class classifier, the implementation is similar to
the binary classifier case, albeit with one important difference:

**Labels are 0-indexed**. The labels/classes need to be take an index [0, K), where K is the number of classes.
{: .notice-warning}

{% highlight scala %}
val training = MLUtils.loadLibSVMFile(sc, covPath)
val solverOptions: SolverOptions[Vector[Double], Double] = new SolverOptions()

val model = MultiClassSVMWithDBCFW.train(training, numClasses, solverOptions)
{% endhighlight %}
