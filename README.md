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
In order to run the examples, the solver needs to built first:
```bash
cd dissolve-struct
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
```
The example can be executed locally as:

```bash
spark-1.1.0/bin/spark-submit --jars dissolve-struct/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar --class "ch.ethz.dal.dissolve.examples.bsvm.COVBinary" --master local dissolve-struct-examples/target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
```

In case this throws an OutOfMemoryError, the executor memory can be increased like so:
```bash
spark-1.1.0/bin/spark-submit --jars dissolve-struct/target/scala-2.10/dissolvestruct_2.10-0.1-SNAPSHOT.jar --class "ch.ethz.dal.dissolve.examples.bsvm.COVBinary" --master local --driver-memory 2G dissolve-struct-examples/target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
```
## Setting up a development environment
To import the packages in Eclipse, the respective .classpath files needs to be generated for the dissolve-struct:
```bash
cd dissolve-struct
sbt eclipse
```
Similarly, for dissolve-struct-examples package too:
```bash
cd dissolve-struct-examples
sbt eclipse
```
The above packages can be imported individually into Eclipse using: File -> Import -> Existing Projects into Workspace

Suppose Eclipse defaults to Scala 2.11, it might issue a "cross-compiled with an incompatible version of Scala".
The correct version needs to be set for both the projects by:
Right clicking on the project -> Scala Compiler -> Setting "Scala Installation" to "Latest 2.10 bundle"
