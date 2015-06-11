'''Builds the lib and examples packages. Then, syncs to Dropbox
'''
import os
import subprocess
import shutil

from paths import PROJECT_DIR

SYNC_JARS = True

HOME_DIR = os.getenv("HOME")

OUTPUT_DIR = os.path.join(HOME_DIR, 'Dropbox/Public/')

LIB_JAR_PATH = os.path.join(HOME_DIR,
                            '.ivy2/local/ch.ethz.dalab/dissolvestruct_2.10/0.1-SNAPSHOT/jars/dissolvestruct_2.10.jar')
EXAMPLES_JAR_PATH = os.path.join(PROJECT_DIR, 'dissolve-struct-examples', 'target/scala-2.10/',
                                 'dissolvestructexample_2.10-0.1-SNAPSHOT.jar')
SCOPT_JAR_PATH = os.path.join(HOME_DIR, '.ivy2/cache/com.github.scopt/scopt_2.10/jars/scopt_2.10-3.3.0.jar')


def execute(command, cwd='.'):
    subprocess.check_call(command, cwd=cwd)


def main():
    dissolve_lib_dir = os.path.join(PROJECT_DIR, 'dissolve-struct-lib')
    dissolve_examples_dir = os.path.join(PROJECT_DIR, 'dissolve-struct-examples')

    # Build lib package
    print "=== Building dissolve-struct-lib ==="
    lib_build_cmd = ["sbt", "publish-local"]
    execute(lib_build_cmd, cwd=dissolve_lib_dir)

    # Build examples package
    print "=== Building dissolve-struct-examples ==="
    examples_build_cmd = ["sbt", "package"]
    execute(examples_build_cmd, cwd=dissolve_examples_dir)


    # Sync all packages to specified output directory
    if SYNC_JARS:
        print "=== Syncing Jars to Dropbox Public Folder ==="
        print LIB_JAR_PATH
        shutil.copy(LIB_JAR_PATH, OUTPUT_DIR)
        print EXAMPLES_JAR_PATH
        shutil.copy(EXAMPLES_JAR_PATH, OUTPUT_DIR)
        print SCOPT_JAR_PATH
        shutil.copy(SCOPT_JAR_PATH, OUTPUT_DIR)


if __name__ == '__main__':
    main()