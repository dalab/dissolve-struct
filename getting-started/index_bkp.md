---
layout: article
title: "Getting Started"
date: 2014-06-25T13:57:25-04:00
modified:
excerpt:
tags: []
image:
  feature:
  teaser:
  thumb:
share: false
---

dissolve<sup>struct</sup> is a distributed solver library for structured output prediction, based on [Apache Spark](http://spark.apache.org).

The library is based on the primal-dual [BCFW solver](http://jmlr.org/proceedings/papers/v28/lacoste-julien13), allowing approximate inference oracles, and distributes this algorithm using the recent communication efficient [CoCoA](http://papers.nips.cc/paper/5599-communication-efficient-distributed-dual-coordinate-ascent) scheme.
The interface to the user is the same as in the widely used [SVM<sup>struct</sup>](http://www.cs.cornell.edu/people/tj/svm_light/svm_struct.html) in the single machine case.


{% include toc.html %}

# Usage

## Obtaining the Software

**Option 1 - Build locally using sbt (Recommended)**

Checkout the project repository

	git clone https://github.com/dalab/dissolve-struct.git

Build the solver package first:
{% highlight bash %}
cd dissolve-struct-lib
sbt publish-local
{% endhighlight %}
Followed by building the example package:
{% highlight bash %}
cd dissolve-struct-examples
sbt package
{% endhighlight %}

(you might have to install `sbt` first, which can be done by running `brew install sbt` on a mac or `sudo apt-get install sbt` on Ubuntu.)

**Option 2 - Obtain packaged binaries**

The binaries for both the solver and the examples package can be obtained at the [releases](https://github.com/dalab/dissolve-struct/releases) page.

But be warned, these binaries might not be up-to-date since the project is still in the development stage.


## Running the examples

### Obtaining the datasets

Obtain the datasets using:

{% highlight bash %}
python helpers/retrieve_datasets.py
{% endhighlight %}

(you might have to `brew install wget` first if on a mac. Additionally, `pip -r requirements.txt` to obtain the python dependencies.)

### Executing through command line

Download the [pre-build binary package of Spark](http://spark.apache.org/downloads.html). Here for example we assume the Spark folder is named `spark-1.X`.

Running the examples follows the format:
{% highlight bash %}
spark-1.X/bin/spark-submit \
	--jars <dissolve-struct-jar-path> \
	--class <class> \
	--master local \
	--driver-memory 2G \
	<example-jar-path>
	<optional-arguments>
{% endhighlight %}

For examples, the Binary classification example can be run within the `dissolve-struct-examples` directory using:
{% highlight bash %}
spark-1.X/bin/spark-submit \
	--jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar \
	--class "ch.ethz.dalab.dissolve.examples.binaryclassification.COVBinary" \
	--master local \
	--driver-memory 2G \
	target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
{% endhighlight %}


### Executing within Eclipse

To ease debugging and development, all examples can also directly be run within Eclipse by `Run As | Scala Application`. This does not require Spark binaries. See the section below how to set up the environment.

Within Eclipse, Spark can only be run in local mode since all the interactions need to be visible to Eclipse.
In order to enable this, the `SparkContext` needs to be initialized by setting the master to `local`:
{% highlight scala %}
val conf = new SparkConf()
	       .setAppName("COV-example")
	       .setMaster("local[4]")
{% endhighlight %}



# Setting up a development environment
We recommend using [Eclipse for Scala](http://scala-ide.org/download/sdk.html), though a similar setup can also be done in Intellij IDEA.
To create an Eclipse Scala project for our purposes, the following simple `sbt` command can be used. This generates the respective `.classpath` files needed.
{% highlight bash %}
cd dissolve-struct-lib
sbt eclipse
{% endhighlight %}
Similarly, for `dissolve-struct-examples` package too:
{% highlight bash %}
cd dissolve-struct-examples
sbt eclipse
{% endhighlight %}
The resulting projects from above can now be imported individually into Eclipse using: `File | Import | Existing Projects into Workspace`. Make sure you have `search for nested projects` checked, so you'll have the choice to select both the `dissolve-struct-lib` and `dissolve-struct-examples` projects, if desired.

Currently Scala 2.10.4 is required by Spark. If Eclipse defaults to Scala 2.11 instead, this can cause an error "cross-compiled with an incompatible version of Scala".
The correct version needs to be set for both the projects by:
`Project Properties | Scala Compiler | Setting "Scala Installation" to "Latest 2.10 bundle"`.
Alternatively, we recommend directly working with Eclipse IDE for Scala 2.10.4 from <http://scala-ide.org/download/sdk.html>.


# Implementing your own Application of Structured Prediction using dissolve<sup>struct</sup>

To implement a custom structured prediction application, using dissolve<sup>struct</sup>, three items are required.
These items are designed to work around any kind of structured objects.
The input and output are represented using Scala generics and are tied to types `X` and `Y` respectively.

## Functions

Similar to [SVM<sup>struct</sup>](http://www.cs.cornell.edu/people/tj/svm_light/svm_struct.html), 3 functions need to be provided:

1. **Feature Function**

   The feature map \\[ \psi: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^d \\] which encodes the input/output pairs.

2. **Loss function**

	The loss function \\( \Delta(Y_i, Y) \\)

3. **Maximization Oracle**

   An oracle which computes the most violating constraint by solving:

	\\[ \hat{Y} = \arg \min_{Y \in \mathcal{Y}_i}  E_w(Y) - \Delta(Y_i, Y) \\]

	where \\( E_w(Y) \\) is the linear classifier function.

4. **Prediction function** (optional)

   A prediction function that computes:

	\\[ \hat{Y} = \arg \min_{Y \in \mathcal{Y}_i} E_w(Y) \\]

	If no prediction function is defined, the solver will call the maximization oracle with the \\(\Delta\\) term set to zero.


These functions need to be implemented by using a class/object which mixes-in the trait `DissolveFunctions`.

{% highlight scala %}
trait DissolveFunctions[X, Y] extends Serializable {

  def featureFn(x: X, y: Y): Vector[Double]

  def lossFn(yPredicted: Y, yTruth: Y): Double

  def oracleFn(model: StructSVMModel[X, Y], x: X, y: Y): Y

  def predictFn(model: StructSVMModel[X, Y], x: X): Y

}
{% endhighlight %}


## Data

The data need to be in the form of an RDD consisting of `LabeledObject[X,Y]` objects.

## Solver Parameters

The parameters for the solver can be set using `SolverOptions`:
{% highlight scala %}
val solverOptions: SolverOptions[X, Y] = new SolverOptions()
solverOptions.lambda = 0.01    // regularization parameter
{% endhighlight %}



With these three items, a model can be trained as follows:

{% highlight scala %}
val trainer: StructSVMWithDBCFW[X, Y] =
	new StructSVMWithDBCFW[X, Y](trainDataRDD,
				ImageSegmentation,
				solverOptions)

val model: StructSVMModel[X, Y] = trainer.trainModel()

{% endhighlight %}


# References
The CoCoA algorithmic framework is described in the following paper:

 * _Jaggi, M., Smith, V., Takac, M., Terhorst, J., Krishnan, S., Hofmann, T., & Jordan, M. I. (2014) [Communication-Efficient Distributed Dual Coordinate Ascent](http://papers.nips.cc/paper/5599-communication-efficient-distributed-dual-coordinate-ascent) (pp. 3068–3076). NIPS 2014 - Advances in Neural Information Processing Systems 27._

  see also the binary classification and regression [spark code here](https://github.com/gingsmith/cocoa).

The (single machine) BCFW algorithm for structured prediction is described in the following paper:

 * _Lacoste-Julien, S., Jaggi, M., Schmidt, M., & Pletscher, P. (2013) [Block-Coordinate Frank-Wolfe Optimization for Structural SVMs](http://jmlr.org/proceedings/papers/v28/lacoste-julien13). ICML 2013 - Proceedings of the 30th International Conference on Machine Learning._

# License

This theme is free and open source software, distributed under the MIT License. So feel free to use this Jekyll theme on your site without linking back to me or including a disclaimer.
