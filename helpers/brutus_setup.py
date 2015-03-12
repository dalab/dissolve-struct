'''Setup dissolve^struct environment on Brutus
'''
__author__ = 'tribhu'

import os
import argparse

from benchmark_utils import ssh

from retrieve_datasets import retrieve, download_to_gen_dir
from paths import PROJECT_DIR

LIB_JAR_URL = 'https://dl.dropboxusercontent.com/u/12851272/dissolvestruct_2.10.jar'
EXAMPLES_JAR_PATH = 'https://dl.dropboxusercontent.com/u/12851272/dissolvestructexample_2.10-0.1-SNAPSHOT.jar'
SCOPT_JAR_PATH = 'https://dl.dropboxusercontent.com/u/12851272/scopt_2.10-3.3.0.jar'


def main():
    parser = argparse.ArgumentParser(description='Setup Brutus cluster for dissolve^struct')
    parser.add_argument("username", help="Username (which has access to Hadoop Cluster)")
    args = parser.parse_args()

    username = args.username

    def ssh_brutus(command):
        ssh('hadoop', username, command)

    # === Obtain data ===
    retrieve(download_all=True)

    # === Obtain the jars ===
    jars_dir = os.path.join(PROJECT_DIR, 'jars')
    if not os.path.exists(jars_dir):
        os.makedirs(jars_dir)

    print "=== Downloading executables to: ", jars_dir, "==="
    for jar_url in [LIB_JAR_URL, EXAMPLES_JAR_PATH, SCOPT_JAR_PATH]:
        print "== Retrieving ", jar_url.split('/')[-1], '=='
        download_to_gen_dir(jar_url, jars_dir)

    # === Move data to HDFS ===
    print "=== Moving data to HDFS ==="
    put_data_cmd = "hadoop fs -put dissolve-struct/data /user/%s/data" % username
    ssh_brutus(put_data_cmd)


if __name__ == '__main__':
    main()