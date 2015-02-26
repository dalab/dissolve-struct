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

(you might have to install `sbt` first, which can be done by`brew install sbt` if on a mac.)

**Option 2 - Obtain packaged binaries**

The binaries for both the solver and the examples package can be obtained at the [releases](https://github.com/dalab/dissolve-struct/releases) page.

But be warned, these binaries might not be up-to-date since the project is still in the development stage.


## Running the examples

### Obtaining the datasets

Obtain the datasets using:

{% highlight bash %}
cd data
bash retrieve_datasets.sh
python convert-ocr-data.py
{% endhighlight %}

(you might have to `brew install wget` first if on a mac. Additionally, `pip -r requirements.txt` to obtain the python dependencies.)

### Executing through command line

Download the [pre-build binary package of Spark](http://spark.apache.org/downloads.html). Here for example we assume the Spark folder is named `spark-1.2.0`.

Running the examples follows the format:
{% highlight bash %}
spark-1.2.0/bin/spark-submit \
	--jars <dissolve-struct-jar-path> \
	--class <class> \
	--master local \
	--driver-memory 2G \
	<example-jar-path>
	<optional-arguments>
{% endhighlight %}

For examples, the Binary classification example can be run within the `dissolve-struct-examples` directory using:
{% highlight bash %}
spark-1.2.0/bin/spark-submit \
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


## Examples

### Binary classification
Training a binary SVM locally from the command-line is done as follows, here for the Forest Cover (COV) dataset. Within `dissolve-struct-examples` directory, run
{% highlight bash %}
spark-1.2.0/bin/spark-submit \
	--jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar \
	--class "ch.ethz.dalab.dissolve.examples.binaryclassification.COVBinary" \
	--master local \
	--driver-memory 2G \
	target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
{% endhighlight %}


### Sequence Prediction with OCR data
![OCR]({{ site.baseurl }}/assets/ocr2.png)

Training a chain structured SVM model on the [OCR dataset](http://www.seas.upenn.edu/~taskar/ocr/). This example uses the Viterbi algorithm for the decoding oracle:
{% highlight bash %}
spark-1.2.0/bin/spark-submit \
	--jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar \
	--class "ch.ethz.dalab.dissolve.examples.chain.ChainDemo" \
	--master local \
	--driver-memory 2G \
	target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
{% endhighlight %}

Here is the same example using more general Belief Propagation, by employing the [Factorie library](http://factorie.cs.umass.edu/) (Requires [Factorie 1.0 Jar](https://github.com/factorie/factorie/releases) to be placed within `dissolve-struct-examples/lib` directory):
{% highlight bash %}
spark-1.2.0/bin/spark-submit \
	--jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar,lib/factorie-1.0.jar \
	--class "ch.ethz.dalab.dissolve.examples.chain.ChainBPDemo" \
	--master local \
	--driver-memory 2G \
	target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
{% endhighlight %}

### Image Segmentation using CRFs
![CRF]({{ site.baseurl }}/assets/imageseg.jpg)

Image Segmentation is performed on the [MSRC](http://research.microsoft.com/en-us/projects/objectclassrecognition/).
This is done by dividing the image into a fixed number of regions and extracting histogram features for each region.
Decoding is performed on a CRF modeled using Factorie, using belief propagation on unary and pairwise features.

This examples requires the dataset (Pixel-wise labelled image v2 dataset) folder downloaded from the MSRC [webpage](http://research.microsoft.com/en-us/projects/objectclassrecognition/) to be placed within the `data/generated` folder.

{% highlight bash %}
spark-1.2.0/bin/spark-submit \
	--jars \ ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar,lib/factorie-1.0.jar \
	--class "ch.ethz.dalab.dissolve.examples.imageseg.ImageSegmentationDemo" \
	--master local \
	--driver-memory 2G \
	target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar \
{% endhighlight %}

The first time the command is executed, the features from the images are pre-computed and stored within the MSRC folder to speed up subsequent executions.
This however make take around 10 minutes, depending on the machine.

Training the model over pairwise factors can be extremely slow, since the maximum a posteriori needs to be computed over thousands of factors, each which can take a combination of 24<sup>2</sup> labels.
The training can be done quickly with only unary features using the `-onlyunaries` flag.



# Setting up a development environment
We recommend using [Eclipse for Scala](http://scala-ide.org/download/sdk.html), though a similar setup can also be done in Intellij IDEA.
To create an Eclipse Scala project for our purposes, the following simple `sbt` command can be used. This generates the respective .classpath files needed.
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

# References
The CoCoA algorithmic framework is described in the following paper:

 * _Jaggi, M., Smith, V., Takac, M., Terhorst, J., Krishnan, S., Hofmann, T., & Jordan, M. I. (2014) [Communication-Efficient Distributed Dual Coordinate Ascent](http://papers.nips.cc/paper/5599-communication-efficient-distributed-dual-coordinate-ascent) (pp. 3068–3076). NIPS 2014 - Advances in Neural Information Processing Systems 27._

  see also the binary classification and regression [spark code here](https://github.com/gingsmith/cocoa).

The (single machine) BCFW algorithm for structured prediction is described in the following paper:

 * _Lacoste-Julien, S., Jaggi, M., Schmidt, M., & Pletscher, P. (2013) [Block-Coordinate Frank-Wolfe Optimization for Structural SVMs](http://jmlr.org/proceedings/papers/v28/lacoste-julien13). ICML 2013 - Proceedings of the 30th International Conference on Machine Learning._
