name := "DissolveStructExample"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.4"

libraryDependencies += "ch.ethz.dal" %% "dissolvestruct" % "0.1-SNAPSHOT"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.1.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.1.0"

resolvers += "IESL Release" at "http://dev-iesl.cs.umass.edu/nexus/content/groups/public"

libraryDependencies += "cc.factorie" % "factorie" % "1.0"
