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

## Checkout the example project in Eclipse
use
Import > New project from git

## Install Spark
Not necessary anymore here, as all dependencies are automatically taken care of by maven.

Old fashioned way: Download spark-1.1.0-bin-hadoop2.4
from here:
<http://spark.apache.org/downloads.html>
and expand the archive in this directory here. This will be the Spark home from now on.
Refresh the project in Eclipse, to recognize the Spark jar dependency.

Right click on you project > project properties > java build path, add the spark .jar as a library (the .jar file in question is 'lib/spark-assembly-1.1.0-hadoop2.4.0.jar')

### How to compile
In a console while in this directory here, run

    sbt package

To run the sample application locally on 4 threads, use

    spark-1.1.0-bin-hadoop2.4/bin/spark-submit   --class "SimpleApp"   --master local[4]   target/scala-2.10/simple-project_2.10-1.0.jar
    
