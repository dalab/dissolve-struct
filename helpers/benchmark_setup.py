"""Prepare the EC2 cluster for dissolve^struct experiments.

    Given a cluster created using spark-ec2 setup, this script will:
    - retrieve the datasets and places them into HDFS
    - build the required packages and place them into appropriate folders
    - setup the execution environment

    Reuses ssh code from AmpLab's Big Data Benchmarks
"""
import argparse
import os

from benchmark_utils import *

WDIR = "/home/ec2-user"  # Working directory


def main():
    parser = argparse.ArgumentParser(description='Setup benchmark cluster')
    parser.add_argument("identity_file", help="SSH private key to log into spark nodes")
    parser.add_argument("master_uri", help="URI of master node")
    args = parser.parse_args()

    master_host = args.master_uri
    identity_file = args.identity_file

    dissolve_dir = os.path.join(WDIR, "dissolve-struct")
    dissolve_lib_dir = os.path.join(dissolve_dir, "dissolve-struct-lib")
    dissolve_examples_dir = os.path.join(dissolve_dir, "dissolve-struct-examples")

    def ssh_spark(command, user="root", cwd=WDIR):
        command = "source /root/.bash_profile; cd %s; %s" % (cwd, command)
        ssh(master_host, user, identity_file, command)

    # === Install all required dependencies ===
    # sbt
    ssh_spark("curl https://bintray.com/sbt/rpm/rpm | sudo tee /etc/yum.repos.d/bintray-sbt-rpm.repo")
    ssh_spark("yum install sbt -y")
    # python pip
    ssh_spark("yum install python27 -y")
    ssh_spark("yum install python-pip -y")

    # === Checkout git repo ===
    ssh_spark("git clone https://github.com/dalab/dissolve-struct.git %s" % dissolve_dir)

    # === Build packages ===
    # Build lib
    # Jar location:
    # /root/.ivy2/local/ch.ethz.dalab/dissolvestruct_2.10/0.1-SNAPSHOT/jars/dissolvestruct_2.10.jar
    ssh_spark("sbt publish-local", cwd=dissolve_lib_dir)
    # Build examples
    # Jar location:
    # /home/ec2-user/dissolve-struct/dissolve-struct-examples/target/scala-2.10/dissolvestructexample_2.10-0.1-SNAPSHOT.jar
    ssh_spark("sbt package", cwd=dissolve_examples_dir)

    # === Data setup ===
    # Install pip dependencies
    ssh_spark("pip install -r requirements.txt", cwd=dissolve_dir)

    # Execute data retrieval script
    ssh_spark("python helpers/retrieve_datasets.py -d", cwd=dissolve_dir)

    # === Move data to HDFS ===
    data_dir = os.path.join(dissolve_dir, "data")
    ssh_spark("/root/ephemeral-hdfs/bin/hadoop fs -put %s data" % data_dir)

    # === Create a file to mark everything is setup ===
    ssh_spark("touch onesmallstep", cwd=WDIR)


if __name__ == '__main__':
    main()
