dissolve<sup>struct</sup>
===========

Distributed solver library for structured output prediction, based on Spark.

The library is based on the primal-dual BCFW solver, allowing approximate inference oracles, and distributes this algorithm using the recent communication efficient CoCoA scheme.
The interface to the user is the same as in the widely used SVM<sup>struct</sup> in the single machine case.

# Usage

This is a standalone spark project, and works with the Scala version of the Eclipse IDE for Scala 2.10.4 (Spark requires this version of Scala currently), download here:
<http://scala-ide.org/download/sdk.html>

Our setup mostly follows the quick-start guide here:
<http://spark.apache.org/docs/latest/quick-start.html#standalone-applications>

## Checkout the project repository

	git clone https://github.com/dalab/dissolve-struct.git

## Running the examples
In order to run the examples, the solver package needs to be built and published locally first:
```bash
cd dissolve-struct-lib
sbt publish-local
```
This is followed by building the example package:
```bash
cd dissolve-struct-examples
sbt package
```

Obtain the datasets by:
```bash
cd data
bash retrieve_datasets.sh
python convert-ocr-data.py
```
(you might have to install `brew install wget` first if on a mac)

#### Executing though command line (Requires Apache Spark Binaries)
Binary classification on the Forest Cover (COV) dataset for example can be executed locally (within `dissolve-struct-examples` directory) on command-line as:
```bash
spark-1.1.0/bin/spark-submit --jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar --class "ch.ethz.dalab.dissolve.examples.bsvm.COVBinary" --master local target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
```

In case this throws an OutOfMemoryError, the executor memory can be increased like so:
```bash
spark-1.1.0/bin/spark-submit --jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar --class "ch.ethz.dalab.dissolve.examples.bsvm.COVBinary" --master local --driver-memory 2G target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
```

[Chain OCR](http://www.seas.upenn.edu/~taskar/ocr/):
```bash
spark-1.1.0/bin/spark-submit --jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar --class "ch.ethz.dalab.dissolve.examples.chain.ChainDemo" --master local --driver-memory 2G target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
```

Chain OCR using [Factorie](http://factorie.cs.umass.edu/) (Requires [Factorie 1.0 Jar](https://github.com/factorie/factorie/releases) to be placed within `dissolve-struct-examples/lib` directory):
```bash
spark-1.1.0/bin/spark-submit --jars ../dissolve-struct-lib/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar,lib/factorie-1.0.jar --class "ch.ethz.dalab.dissolve.examples.chain.ChainBPDemo" --master local --driver-memory 2G target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
```

#### Executing within Eclipse
To ease debugging and development, the examples can be run within Eclipse by `Run As | Scala Application`.
However, this can be run only in local mode since all the interactions needs to be visible to Eclipse.
In order to enable this, the `SparkContext` needs to be initialized by setting the master to `local`:
```scala
val conf = new SparkConf()
	       .setAppName("COV-example")
	       .setMaster("local[4]")
```

## Setting up a development environment
To import the packages in Eclipse, the respective .classpath files needs to be generated for the `dissolve-struct-lib`:
```bash
cd dissolve-struct-lib
sbt eclipse
```
Similarly, for `dissolve-struct-examples` package too:
```bash
cd dissolve-struct-examples
sbt eclipse
```
The above packages can be imported individually into Eclipse using: `File | Import | Existing Projects into Workspace`

Suppose Eclipse defaults to Scala 2.11, it might issue a "cross-compiled with an incompatible version of Scala".
The correct version needs to be set for both the projects by:
`Project Properties | Scala Compiler | Setting "Scala Installation" to "Latest 2.10 bundle"`
