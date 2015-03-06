"""Retrieves datasets from various sources
"""
import urllib
import subprocess
from paths import *


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

def main():
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
    for url in CHAIN_URLS:
        download_to_gen_dir(url)


if __name__ == '__main__':
    main()
