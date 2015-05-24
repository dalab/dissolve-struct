---
layout: article
title: "Semantic Image Segmentation With CRFs"
modified:
categories: examples
excerpt:
tags: []
image:
  feature: imageseg_full.jpg
  teaser: imageseg_thumb.png
  thumb: imageseg_thumb.png
date: 2015-05-24T22:58:03+02:00
---


Image Segmentation is performed on the [MSRC](http://research.microsoft.com/en-us/projects/objectclassrecognition/).
This is done by dividing the image into a fixed number of regions and extracting histogram features for each region.
Decoding is performed on a CRF modeled using Factorie, using belief propagation on unary and pairwise features.

This examples requires the dataset (Pixel-wise labelled image v2 dataset) downloaded from the MSRC [webpage](http://research.microsoft.com/en-us/projects/objectclassrecognition/) to be placed within the `data/generated` directory.

{% highlight bash %}
spark-1.X/bin/spark-submit \
	--jars \ ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar,lib/factorie-1.0.jar \
	--class "ch.ethz.dalab.dissolve.examples.imageseg.ImageSegmentationDemo" \
	--master local \
	--driver-memory 2G \
	target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar \
{% endhighlight %}

The first time the command is executed, the features from the images are pre-computed and stored within the MSRC directory to speed up subsequent executions.
This however make take around 10 minutes, depending on the machine.

Training the model over pairwise factors can be extremely slow, since the MAP assignment needs to be computed over thousands of factors, each which can take a combination of 24<sup>2</sup> labels.
The training can be done quickly with only unary features using the `-onlyunaries` flag.
