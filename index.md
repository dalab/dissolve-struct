---
layout: dis_template
---

dissolve<sup>struct</sup> is a Distributed solver library for structured output prediction, based on Spark.

The library is based on the primal-dual BCFW solver, allowing approximate inference oracles, and distributes this algorithm using the recent communication efficient CoCoA scheme.
The interface to the user is the same as in the widely used SVM<sup>struct</sup> in the single machine case.

<h1> Table of Contents </h1>
* auto-gen TOC:
{:toc}

# Usage

## Checkout the project repository

	git clone https://github.com/dalab/dissolve-struct.git

## Running the examples

### Obtain data

Obtain the datasets using:

{% highlight bash %}
cd data
bash retrieve_datasets.sh
python convert-ocr-data.py
{% endhighlight %}

(you might have to install `brew install sbt` and `brew install wget` first if on a mac. Additionally, `pip -r requirements.txt` to obtain the python dependencies.)

The library and examples packages need to be built and packaged to execute the examples.
This can be done in two ways.

### Obtain binaries

**Option 1 - Build locally using sbt (Recommended)**

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

**Option 2 - Obtain packaged binaries**

The binaries for both the solver and the examples package can be obtained at the [releases](https://github.com/dalab/dissolve-struct/releases) page.

But be warned, these binaries might not be up-to-date since the project is still in the development stage.

### Execution

**Executing through command line**

Download the [pre-build binary package of Spark](http://spark.apache.org/downloads.html). Here for example we assume the Spark folder is named `spark-1.2.0`.

##### Binary classification Example
Training a binary SVM locally from the command-line is done as follows, here for the Forest Cover (COV) dataset. Within `dissolve-struct-examples` directory, run
{% highlight bash %}
spark-1.2.0/bin/spark-submit --jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar --class "ch.ethz.dalab.dissolve.examples.binaryclassification.COVBinary" --master local --driver-memory 2G target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
{% endhighlight %}

##### Sequence Prediction with OCR data
Training a chain structured SVM model on the [OCR dataset](http://www.seas.upenn.edu/~taskar/ocr/). This example uses the Viterbi algorithm for the decoding oracle:
{% highlight bash %}
spark-1.2.0/bin/spark-submit --jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar --class "ch.ethz.dalab.dissolve.examples.chain.ChainDemo" --master local --driver-memory 2G target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
{% endhighlight %}

Here is the same example using more general Belief Propagation, by employing the [Factorie library](http://factorie.cs.umass.edu/) (Requires [Factorie 1.0 Jar](https://github.com/factorie/factorie/releases) to be placed within `dissolve-struct-examples/lib` directory):
{% highlight bash %}
spark-1.2.0/bin/spark-submit --jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar,lib/factorie-1.0.jar --class "ch.ethz.dalab.dissolve.examples.chain.ChainBPDemo" --master local --driver-memory 2G target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
{% endhighlight %}

**Executing within Eclipse**

To ease debugging and development, the examples can directly be run within Eclipse by `Run As | Scala Application`. This does not require Spark binaries. See the section below how to set up the environment.

Within Eclipse, Spark can only be run in local mode since all the interactions need to be visible to Eclipse.
In order to enable this, the `SparkContext` needs to be initialized by setting the master to `local`:
{% highlight scala %}
val conf = new SparkConf()
	       .setAppName("COV-example")
	       .setMaster("local[4]")
{% endhighlight %}

# Setting up a development environment
To import the packages in [Eclipse for Scala](http://scala-ide.org/download/sdk.html), the respective .classpath files needs to be generated for the `dissolve-struct-lib`:
{% highlight bash %}
cd dissolve-struct-lib
sbt eclipse
{% endhighlight %}
Similarly, for `dissolve-struct-examples` package too:
{% highlight bash %}
cd dissolve-struct-examples
sbt eclipse
{% endhighlight %}
The above packages can be imported individually into Eclipse using: `File | Import | Existing Projects into Workspace`. Make sure you have `search for nested projects` checked, so you'll have the choice to select both the `dissolve-struct-lib` and `dissolve-struct-examples` projects, if desired.

Currently Scala 2.10.4 is required by Spark. If Eclipse defaults to Scala 2.11 instead, this can cause an error "cross-compiled with an incompatible version of Scala".
The correct version needs to be set for both the projects by:
`Project Properties | Scala Compiler | Setting "Scala Installation" to "Latest 2.10 bundle"`.
Alternatively, we recommend directly working with Eclipse IDE for Scala 2.10.4 from <http://scala-ide.org/download/sdk.html>.

# References
The CoCoA algorithmic framework is described in the following paper:

 * _Jaggi, M., Smith, V., Takac, M., Terhorst, J., Krishnan, S., Hofmann, T., & Jordan, M. I. (2014) [Communication-Efficient Distributed Dual Coordinate Ascent](http://papers.nips.cc/paper/5599-communication-efficient-distributed-dual-coordinate-ascent) (pp. 3068–3076). NIPS 2014 - Advances in Neural Information Processing Systems 27._

  see also the binary classification and regression [spark code here](https://github.com/gingsmith/cocoa).

The (single machine) BCFW algorithm for structured prediction is described in the following paper:

 * _Lacoste-Julien, S., Jaggi, M., Schmidt, M., & Pletscher, P. (2013) [Block-Coordinate Frank-Wolfe Optimization for Structural SVMs](http://jmlr.org/proceedings/papers/v28/lacoste-julien13). ICML 2013 - Proceedings of the 30th International Conference on Machine Learning._



