name := "DissolveStruct"

organization := "ch.ethz.dalab"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.3.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.3.0"

libraryDependencies += "org.scalanlp" %% "breeze" % "0.11.1"

libraryDependencies += "org.scalanlp" %% "breeze-natives" % "0.11.1"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0"

resolvers += Resolver.sonatypeRepo("public")
