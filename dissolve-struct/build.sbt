name := "DissolveStruct"

organization := "ch.ethz.dal"

version := "0.1-SNAPSHOT"

scalaVersion := "2.10.4"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.1.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.1.0"

libraryDependencies += "org.scalanlp" %% "breeze" % "0.10"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

retrieveManaged := true

unmanagedBase := baseDirectory.value / "lib_managed"

unmanagedJars in Compile := (baseDirectory.value ** "*.jar").classpath

unmanagedJars in Compile ++= {
	val base = baseDirectory.value
	val baseDirectories = base / "lib_managed"
	val customJars = baseDirectories ** "*.jar"
	customJars.classpath
}

resolvers += "IESL Release" at "http://dev-iesl.cs.umass.edu/nexus/content/groups/public"

// libraryDependencies += "cc.factorie" % "factorie" % "1.0"


