"""Retrieves datasets from various sources
"""
import urllib
import subprocess
from paths import *
import argparse

from ocr_helpers import convert_ocr_data


def decompress(filename):
    print "Decompressing: ", filename
    try:
        subprocess.check_call(['bzip2', '-d', filename])
    except subprocess.CalledProcessError as e:
        pass


def download_to_gen_dir(url):
    print "Downloading: ", url
    basename = os.path.basename(url)
    destname = os.path.join(GEN_DATA_DIR, basename)
    urllib.urlretrieve(url, destname)
    return destname


def download_and_decompress(url):
    destname = download_to_gen_dir(url)
    decompress(destname)


def retrieve(download_all=False):
    # Retrieve the files
    print "=== A1A ==="
    download_to_gen_dir(A1A_URL)

    print "=== COV BINARY ==="
    download_and_decompress(COV_BIN_URL)

    print "=== COV MULTICLASS ==="
    download_and_decompress(COV_MULT_URL)

    print "=== RCV1 ==="
    download_and_decompress(RCV1_URL)

    print "=== CHAIN ==="
    if download_all:
        for url in CHAIN_URLS:
            download_to_gen_dir(url)
    else:
        convert_ocr_data()


def main():
    parser = argparse.ArgumentParser(description='Retrieve datasets for dissolve^struct')
    parser.add_argument("-d", "--download", action="store_true",
                        help="Download files instead of processing when possible")
    args = parser.parse_args()

    retrieve(args.download)


if __name__ == '__main__':
    main()
