'''Setup dissolve^struct environment on Brutus
'''
__author__ = 'tribhu'

import os

from retrieve_datasets import retrieve, download_to_gen_dir
from paths import PROJECT_DIR

LIB_JAR_URL = 'https://dl.dropboxusercontent.com/u/12851272/dissolvestruct_2.10.jar'
EXAMPLES_JAR_PATH = 'https://dl.dropboxusercontent.com/u/12851272/dissolvestructexample_2.10-0.1-SNAPSHOT.jar'
SCOPT_JAR_PATH = 'https://dl.dropboxusercontent.com/u/12851272/scopt_2.10-3.3.0.jar'


def main():
    # === Obtain data ===
    retrieve(download_all=True)

    # === Obtain the jars ===
    jars_dir = os.path.join(PROJECT_DIR, 'jars')
    if not os.path.exists(jars_dir):
        os.makedirs(jars_dir)

    print "Downloading executables to: ", jars_dir
    for jar_url in [LIB_JAR_URL, EXAMPLES_JAR_PATH, SCOPT_JAR_PATH]:
        print "== Retrieving ", jar_url.split('/')[-1], '=='
        download_to_gen_dir(jar_url, jars_dir)


if __name__ == '__main__':
    main()