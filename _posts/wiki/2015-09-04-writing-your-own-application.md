---
layout: article
title: "Writing Your Own Application"
modified:
categories: wiki
excerpt:
tags: [wiki, tutorial]
image:
  feature:
  teaser:
  thumb:
date: 2015-09-04T12:07:11+02:00
---

Here, we'll show you how to write your own Distributed Structured Prediction application with
the help of the packaged starter-kit found in `dissolve-struct-application`.
To begin, you'll need:

* Basic experience with [Structured SVMs](https://en.wikipedia.org/wiki/Structured_support_vector_machine)
* Linux or Mac OS X
* [Scala Eclipse IDE](http://scala-ide.org/)
* sbt (`sudo apt-get install sbt` on Ubuntu or `brew install sbt` on OS X)

# Obtain Project repository

First, you'll need to obtain the project repository.

{% highlight bash %}
$ git clone https://github.com/dalab/dissolve-struct.git
{% endhighlight %}

# Setup development environment

Now, we'll need to build dissolve<sup>struct</sup> and generate some files,
so that the project can be imported into Eclipse.

{% highlight bash %}
$ cd dissolve-struct-lib
$ sbt publish-local
$ cd ../dissolve-struct-application
$ sbt eclipse
{% endhighlight %}

The application can now be easily imported into Eclipse via
`File | Import | General | Existing Projects into Workspace`.
Following this, if Eclipse displays build errors such as the library being
cross-compiled with Scala version 2.11, you'll need to go to
`Project Properties | Scala Compiler` by right-clicking on the project and
switch to `Fixed Scala Installation: 2.10.x (built-in)`.

# Implement Dissolve Functions

Now that the development environment is set, you'll need to write define
some implementations for an interface provided using `DissolveFunctions`.
In order to bootstrap and get you started, you'll find a skeleton in
`src/main/scala/ch/ethz/dalab/dissolve/app/DSApp.scala`.
This file contains all the necessary instructions to get you started with
implementing you application.

The main idea is to implement:

1. The Joint feature map \\( \phi \\)
2. A Loss function \\( \Delta \\)
3. The Maximization Oracle \\( H(w) \\)
and provide the training data in the `main()` function.

# Building

While you can test your application locally within Eclipse, the purpose of
the library is to run a scalable application (whose data may be not fit on a single
machine) on a cluster.
For this, you'll need to build your application into a binary jar and hand it
to [spark-submit](http://spark.apache.org/docs/latest/submitting-applications.html).

## Packaging into a jar
First, you'll need to set the metadata and additional libraries used in the
`dissolve-struct-application/build.sbt` file.
This file also contains the instructions which will help you get started.
Don't worry -- this is very straight-forward and you'll merely need to change
a few lines.

After you've got the `build.sbt` configured, you can obtain the fat jar using

{% highlight bash %}
$ cd dissolve-struct-application
$ sbt assembly
{% endhighlight %}

## Running on EC2
It's extremely easy to execute your application on a cluster!
You'll need to first launch an EC2 cluster configured with Spark.
Luckily, Spark contains a script which completely sets it up for you.
In case you've downloaded Spark on your machine, you'll find this script in
`$SPARK_ROOT/ec2/spark-ec2`.
The documentation for this can be found
[here](http://spark.apache.org/docs/latest/ec2-scripts.html).

Once the cluster is setup, you'll need to merely move your jar and data to the master node and
start the application via [spark-submit](http://spark.apache.org/docs/latest/submitting-applications.html).
Also, make sure you're not running in local-mode when submitting the application
(i.e, you don't have a setMaster("local") enabled in your driver code.)

# Java Implementation
If you're a hardcore Java programmer, worry not, dissolve<sup>struct</sup>
applications can be written in Java too.
Some samples can be found
[here](https://bitbucket.org/tribhuvanesh/java-dissolve-struct/overview).
Unfortunately, we are not focusing much on this, but feel free to write
to us or raise an issue on Github if you need help.
