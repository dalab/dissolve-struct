// ---- < 1 > -----------------------------------------------------------------
// Enter your application name, organization and version below
// This information will be used when the binary jar is packaged
name := "DissolveStructApplication"

organization := "ch.ethz.dalab"

version := "0.1-SNAPSHOT" // Keep this unchanged for development releases
// ---- </ 1 > ----------------------------------------------------------------

scalaVersion := "2.10.4"

libraryDependencies += "ch.ethz.dalab" %% "dissolvestruct" % "0.1-SNAPSHOT"

libraryDependencies += "org.scalatest" % "scalatest_2.10" % "2.0" % "test"

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.4.1"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.4.1"

resolvers += "IESL Release" at "http://dev-iesl.cs.umass.edu/nexus/content/groups/public"

libraryDependencies += "cc.factorie" % "factorie" % "1.0"

libraryDependencies += "com.github.scopt" %% "scopt" % "3.3.0"

// ---- < 2 > -----------------------------------------------------------------
// Add additional dependencies in the space provided below, like above.
// Libraries often provide the exact line that needs to be added here on their
// webpage.
// PS: Keep your eyes peeled -- there is a difference between "%%" and "%"

// libraryDependencies += "organization" %% "application_name" % "version"

// ---- </ 2 > ----------------------------------------------------------------

resolvers += Resolver.sonatypeRepo("public")

EclipseKeys.createSrc := EclipseCreateSrc.Default + EclipseCreateSrc.Resource

mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
    {
        case PathList("javax", "servlet", xs @ _*)           => MergeStrategy.first
        case PathList(ps @ _*) if ps.last endsWith ".html"   => MergeStrategy.first
        case "application.conf"                              => MergeStrategy.concat
        case "reference.conf"                                => MergeStrategy.concat
        case "log4j.properties"                              => MergeStrategy.discard
        case m if m.toLowerCase.endsWith("manifest.mf")      => MergeStrategy.discard
        case m if m.toLowerCase.matches("meta-inf.*\\.sf$")  => MergeStrategy.discard
        case _ => MergeStrategy.first
    }
}

test in assembly := {}
