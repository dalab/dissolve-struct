name := "DissolveStructExample"

organization := "ch.ethz.dalab"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.4"

libraryDependencies += "ch.ethz.dalab" %% "dissolvestruct" % "0.1-SNAPSHOT"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.2.1"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.2.1"

resolvers += "IESL Release" at "http://dev-iesl.cs.umass.edu/nexus/content/groups/public"

libraryDependencies += "cc.factorie" % "factorie" % "1.0"
