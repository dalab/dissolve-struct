---
layout: article
title: "Getting Started"
date: 2015-05-24T13:57:25-04:00
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


## Obtaining dissolve<sup>struct</sup>
Checkout the project repository
{% highlight bash %}
$ git clone https://github.com/dalab/dissolve-struct.git
{% endhighlight %}
Build the solver package first:
{% highlight bash %}
$ cd dissolve-struct-lib
$ sbt publish-local
{% endhighlight %}
Followed by building the example package (fat jar):
{% highlight bash %}
$ cd dissolve-struct-examples
$ sbt assembly
{% endhighlight %}

The fat jar produced by this build is placed in
`target/scala-2.10/DissolveStructExample-assembly-0.1-SNAPSHOT.jar` of the
`dissolve-struct-examples` directory.
We'll now refer to this as `<examples-jar-path>`.

(you might have to install `sbt` first, which can be done by running `brew install sbt` on a mac or `sudo apt-get install sbt` on Ubuntu, or follow the instructions available [here](http://www.scala-sbt.org/download.html))

**Alternative:** The binaries for both the solver and the examples package can be obtained at the [releases](https://github.com/dalab/dissolve-struct/releases) page.
But be warned, these binaries might not be up-to-date since the project is still in the development stage.
{: .notice-info}


## Running the examples

##### Obtain datasets
Obtain the datasets using:

{% highlight bash %}
$ python helpers/retrieve_datasets.py
{% endhighlight %}

(you might have to `brew install wget` first if on a mac. Additionally, `pip -r requirements.txt` to obtain the python dependencies.)

##### Executing through command line

Running the examples _locally_ follows the format:
{% highlight bash %}
spark-1.X/bin/spark-submit \
	--class <class> \
	--master local \
	--driver-memory 2G \
	<examples-jar-path>
	<optional-arguments>
{% endhighlight %}

**PS:**  We recommend using the latest Spark version.
{: .notice-info}

For example, the Binary classification example can be run within the `dissolve-struct-examples` directory using:
{% highlight bash %}
spark-1.X/bin/spark-submit \
	--class "ch.ethz.dalab.dissolve.examples.binaryclassification.COVBinary" \
	--master local \
	--driver-memory 2G \
	target/scala-2.10/DissolveStructExample-assembly-0.1-SNAPSHOT.jar
{% endhighlight %}



##### Executing within Eclipse

To ease debugging and development, all examples can also directly be run within Eclipse by `Run As | Scala Application`. This does not require Spark binaries. See the section below how to set up the environment.

Within Eclipse, Spark can only be run in local mode since all the interactions need to be visible to Eclipse.
In order to enable this, the `SparkContext` needs to be initialized by setting the master to `local`:
{% highlight scala %}
val conf = new SparkConf()
	       .setAppName("COV-example")
	       .setMaster("local[4]")
{% endhighlight %}

## Setting up a development environment

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

## Implementing your own application


This section assumes the reader's familiarity with Structured SVMs and the notation.
Please refer to the BCFW paper in the [references](#references) section to learn more.

dissolve<sup>struct</sup> is designed to solve most generic structured prediction
tasks.
In order to do this, the solver requires 3 items:

##### 1. Functions
Our oracle-based solver requires 4 functions, which follow the same interface as
[SVM<sup>struct</sup>](http://www.cs.cornell.edu/people/tj/svm_light/svm_struct.html).

1. **Joint Feature Map**

   The joint feature map
   \\[ \phi: \mathcal{X} \times \mathcal{Y} \rightarrow \mathbb{R}^d \\]
   which encodes the input/output pairs.

2. **Structured Loss**

	The structured loss function \\( \Delta(Y_i, Y) \\), over which the Bayes risk is minimized.

3. **Maximization Oracle**

   An oracle which computes the _most violating constraint_ by solving:
	\\[ \hat{Y} = \arg \max_{Y \in \mathcal{Y}_i} \Delta(Y_i, Y) - \langle w, \psi_i(X_i, Y) \rangle \\]
  where
  \\[ \psi_i(X_i, Y) = \phi(X_i, Y_i) - \phi(X_i, Y) \\]

4. **Prediction function** (optional)

   A prediction function that computes:

	\\[ \hat{Y} = \arg \max_{Y \in \mathcal{Y}_i} \langle w, \phi(X_i, Y) \rangle \\]

	If no prediction function is defined, the solver will call the maximization oracle with the \\(\Delta\\) term set to zero.

These 4 functions need to be implemented by using a class/object which mixes-in the trait `DissolveFunctions`.

{% highlight scala %}
trait DissolveFunctions[X, Y] extends Serializable {

  def featureFn(x: X, y: Y): Vector[Double]

  def lossFn(yPredicted: Y, yTruth: Y): Double

  def oracleFn(model: StructSVMModel[X, Y], x: X, y: Y): Y

  def predictFn(model: StructSVMModel[X, Y], x: X): Y

}
{% endhighlight %}

##### 2. Data

The data need to be in the form of an RDD consisting of `LabeledObject[X,Y]` objects.
Refer to `ch.ethz.dalab.dissolve.regression.LabeledObject` for the definition.

##### 3. Parameters

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


## Feedback

We'd love to hear back on what you think, your experience or any remark
on dissolve<sup>struct</sup>.
You can contact the authors [Martin](http://people.inf.ethz.ch/jaggim/), [Aurelien](http://people.inf.ethz.ch/alucchi/) or [Tribhuvanesh](http://tribhuvanesh.github.io/).


## References
The CoCoA algorithmic framework is described in the following paper:

 * _Jaggi, M., Smith, V., Takac, M., Terhorst, J., Krishnan, S., Hofmann, T., & Jordan, M. I. (2014) [Communication-Efficient Distributed Dual Coordinate Ascent](http://papers.nips.cc/paper/5599-communication-efficient-distributed-dual-coordinate-ascent) (pp. 3068–3076). NIPS 2014 - Advances in Neural Information Processing Systems 27._

  see also the binary classification and regression [spark code here](https://github.com/gingsmith/cocoa).

The (single machine) BCFW algorithm for structured prediction is described in the following paper:

 * _Lacoste-Julien, S., Jaggi, M., Schmidt, M., & Pletscher, P. (2013) [Block-Coordinate Frank-Wolfe Optimization for Structural SVMs](http://jmlr.org/proceedings/papers/v28/lacoste-julien13). ICML 2013 - Proceedings of the 30th International Conference on Machine Learning._

## License

This library is free and distributed under the Apache 2.0 License.
