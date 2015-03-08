"""Keeps track of paths used by other submodules
"""
import os

current_file_path = os.path.abspath(__file__)
PROJECT_DIR = os.path.join(os.path.dirname(current_file_path), os.pardir)

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
GEN_DATA_DIR = os.path.join(DATA_DIR, 'generated')

CHAIN_OCR_FILE = os.path.join(DATA_DIR, "ocr.mat")



# URLS for DATASETS
A1A_URL = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a"

COV_BIN_URL = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2"

COV_MULT_URL = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/covtype.scale.bz2"

RCV1_URL = "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2"

chain_files = ["folds_test.csv",
               "folds_train.csv",
               "patterns_train.csv",
               "patterns_test.csv",
               "labels_train.csv",
               "labels_test.csv",
               ]
s3_chain_base_url = "https://s3-eu-west-1.amazonaws.com/dissolve-struct/chain"
CHAIN_URLS = [os.path.join(s3_chain_base_url, fname) for fname in chain_files]

