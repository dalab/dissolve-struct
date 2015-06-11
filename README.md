DBCFWstruct
===========

A distributed implementation of Block-Coordinate Frank-Wolfe Algorithm for Structural SVMs using Spark


# Usage

This is a standalone spark project, and works with the Scala version of the Eclipse IDE for Scala 2.10.4 (Spark requires this version of Scala currently).
<http://scala-ide.org/download/sdk.html>

Our setup mostly follows the quick-start guide here:
<http://spark.apache.org/docs/latest/quick-start.html#standalone-applications>

## Checkout the example project in Eclipse
use
Import > New project from git

## Install Spark
(The Spark distribution is not included on git here since the .jar file is over 100MB)

Download spark-1.1.0-bin-hadoop2.4
from here:
<http://spark.apache.org/downloads.html>
and expand the archive in this directory here. This will be the Spark home from now on.
Refresh the project in Eclipse, to recognize the Spark jar dependency.

Right click on you project > project properties > java build path, add the spark .jar as a library (the .jar file in question is 'lib/spark-assembly-1.1.0-hadoop2.4.0.jar')

## How to compile
In a console while in this directory here, run

    sbt package

To run the sample application locally on 4 threads, use

    spark-1.1.0-bin-hadoop2.4/bin/spark-submit   --class "SimpleApp"   --master local[4]   target/scala-2.10/simple-project_2.10-1.0.jar
    
