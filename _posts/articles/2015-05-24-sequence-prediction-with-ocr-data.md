---
layout: article
title: "Sequence Prediction With OCR Data"
modified:
categories: examples
excerpt:
tags: []
image:
  feature: ocr_full.png
  teaser: ocr_thumb.png
  thumb: ocr_thumb.png
date: 2015-05-24T22:57:42+02:00
---

Training a chain structured SVM model on the [OCR dataset](http://www.seas.upenn.edu/~taskar/ocr/). This example uses the Viterbi algorithm for the decoding oracle:
{% highlight bash %}
spark-1.X/bin/spark-submit \
	--class "ch.ethz.dalab.dissolve.examples.chain.ChainDemo" \
	--master local \
	--driver-memory 2G \
	<examples-jar-path>
{% endhighlight %}

Here is the same example using more general Belief Propagation, by employing the [Factorie library](http://factorie.cs.umass.edu/) (Requires [Factorie 1.0 Jar](https://github.com/factorie/factorie/releases) to be placed within `dissolve-struct-examples/lib` directory):
{% highlight bash %}
spark-1.X/bin/spark-submit \
	--class "ch.ethz.dalab.dissolve.examples.chain.ChainBPDemo" \
	--master local \
	--driver-memory 2G \
  <examples-jar-path>
{% endhighlight %}
