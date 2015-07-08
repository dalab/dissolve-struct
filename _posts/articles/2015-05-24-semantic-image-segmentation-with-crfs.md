---
layout: article
title: "Semantic Image Segmentation With CRFs"
modified: 2015-07-08
categories: examples
excerpt:
tags: []
image:
  feature: imageseg_full.jpg
  teaser: imageseg_thumb.png
  thumb: imageseg_thumb.png
date: 2015-05-24T22:58:03+02:00
---


Image Segmentation is performed on the [MSRC-21](http://research.microsoft.com/en-us/projects/objectclassrecognition/) dataset.
To use this, you'll need to download and extract the MSRC dataset from [here](https://s3-eu-west-1.amazonaws.com/dissolve-struct/msrc/msrc.tar.gz)
into the `data/generated` directory.
This dataset contains the MSRC-21 labelled images, the extracted super-pixels using [SLIC](http://ivrl.epfl.ch/research/superpixels)
and its [features](http://cvlab.epfl.ch/data/dpg/index.php).

The problem is formulated as CRF parameter learning.
We use simple histogram features for the data term (unaries) and pairwise
transitions.
The maximization oracle performs loss-augmented decoding on a second-order
factor graph using Loopy Belief Propagation through [Factorie](http://factorie.cs.umass.edu/).

In order to run the example, you'll need to execute the following command, from
within the `dissolve-struct-examples` directory.
{% highlight bash %}
spark-1.X/bin/spark-submit \
  --class "ch.ethz.dalab.dissolve.examples.imageseg.ImageSegRunner" \
  --master local \
  --driver-memory 2G \
  <examples-jar-path>
{% endhighlight %}
This should take about 5-6 hours to obtain satisfactory results.

Alternatively, this can be run either on a subset of the original dataset, like so:
{% highlight bash %}
spark-1.X/bin/spark-submit \
  --class "ch.ethz.dalab.dissolve.examples.imageseg.ImageSegRunner" \
  --master local \
  --driver-memory 2G \
  <examples-jar-path> \
  --kwargs train=Train-small.txt,validation=Validation-small.txt
{% endhighlight %}
or by using only the unary features:
{% highlight bash %}
spark-1.X/bin/spark-submit \
  --class "ch.ethz.dalab.dissolve.examples.imageseg.ImageSegRunner" \
  --master local \
  --driver-memory 2G \
  <examples-jar-path> \
  --kwargs unaries=true
{% endhighlight %}
Both of these should finish execution in under 5 minutes.
