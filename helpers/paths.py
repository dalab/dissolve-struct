"""Keeps track of paths used by other submodules
"""
import os

current_file_path = os.path.abspath(__file__)
PROJECT_DIR = os.path.join(os.path.dirname(current_file_path), os.pardir)

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
GEN_DATA_DIR = os.path.join(DATA_DIR, 'generated')

CHAIN_OCR_FILE = os.path.join(DATA_DIR, "ocr.mat")

# If an sbt build cannot be performed, jars can be accessed here
JARS_DIR = os.path.join(PROJECT_DIR, 'jars')
LIB_JAR_PATH = os.path.join(JARS_DIR, 'dissolvestruct_2.10.jar')
EXAMPLES_JAR_PATH = os.path.join(JARS_DIR, 'dissolvestructexample_2.10-0.1-SNAPSHOT.jar')
SCOPT_JAR_PATH = os.path.join(JARS_DIR, 'scopt_2.10-3.3.0.jar')

# Output dir. Any output produced in a subdirectory within this folder.
EXPT_OUTPUT_DIR = os.path.join(PROJECT_DIR, 'benchmark-data')



# URLS for DATASETS
A1A_URL = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a"

COV_BIN_URL = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2"

COV_MULT_URL = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/covtype.scale.bz2"

RCV1_URL = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2"

MSRC_URL = "https://s3-eu-west-1.amazonaws.com/dissolve-struct/msrc/msrc.tar.gz"

chain_files = ["folds_test.csv",
               "folds_train.csv",
               "patterns_train.csv",
               "patterns_test.csv",
               "labels_train.csv",
               "labels_test.csv",
               ]
s3_chain_base_url = "https://s3-eu-west-1.amazonaws.com/dissolve-struct/chain"
CHAIN_URLS = [os.path.join(s3_chain_base_url, fname) for fname in chain_files]

