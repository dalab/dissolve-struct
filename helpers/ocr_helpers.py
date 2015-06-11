import numpy as np
import scipy.io
import os
from paths import *

FILTER_FOLD = True
TST_FOLD_NUMS = [0,] # Datapoints from these folds will be treated as test dataset

def convert_ocr_data():
    idx_trn = 0
    idx_tst = 0

    ocr_mat_path = os.path.join(DATA_DIR, 'ocr.mat')
    patterns_train_path = os.path.join(GEN_DATA_DIR, 'patterns_train.csv')
    patterns_test_path = os.path.join(GEN_DATA_DIR, 'patterns_test.csv')
    labels_train_path = os.path.join(GEN_DATA_DIR, 'labels_train.csv')
    labels_test_path = os.path.join(GEN_DATA_DIR, 'labels_test.csv')
    folds_train_path = os.path.join(GEN_DATA_DIR, 'folds_train.csv')
    folds_test_path = os.path.join(GEN_DATA_DIR, 'folds_test.csv')

    print "Processing features available in %s" % ocr_mat_path

    mat = scipy.io.loadmat(ocr_mat_path, struct_as_record=False, squeeze_me=True)
    n = np.shape(mat['dataset'])[0]
    with open(patterns_train_path, 'w') as fpat_trn, open(labels_train_path, 'w') as flab_trn, open(folds_train_path, 'w') as ffold_trn, \
         open(patterns_test_path, 'w')  as fpat_tst, open(labels_test_path, 'w')  as flab_tst, open(folds_test_path, 'w') as ffold_tst:
        for i in range(n):
            ### Write folds
            fold = mat['dataset'][i].__dict__['fold']
            if fold in TST_FOLD_NUMS:
                fpat, flab, ffold = fpat_tst, flab_tst, ffold_tst
                idx_tst += 1
                idx = idx_tst
            else:
                fpat, flab, ffold = fpat_trn, flab_trn, ffold_trn
                idx_trn += 1
                idx = idx_trn
            # FORMAT: id,fold
            ffold.write('%d,%d\n' % (idx, fold))

            ### Write patterns (x_i's)
            pixels = mat['dataset'][i].__dict__['pixels']
            num_letters = np.shape(pixels)[0]
            letter_shape = np.shape(pixels[0])
            # Create a matrix of size num_pixels+bias_var x num_letters
            xi = np.zeros((letter_shape[0] * letter_shape[1] + 1, num_letters))
            for letter_id in range(num_letters):
                letter = pixels[letter_id] # Returns a 16x8 matrix
                xi[:, letter_id] = np.append(letter.flatten(order='F'), [1.])
            # Vectorize the above matrix and store it
            # After flattening, order is column-major
            xi_str = ','.join([`s` for s in xi.flatten('F')])
            # FORMAT: id,#rows,#cols,x_0_0,x_0_1,...x_n_m
            fpat.write('%d,%d,%d,%s\n' % (idx, np.shape(xi)[0], np.shape(xi)[1], xi_str))

            ### Write labels (y_i's)
            labels = mat['dataset'][i].__dict__['word']
            labels_str = ','.join([`a` for a in labels])
            # FORMAT: id,#letters,letter_0,letter_1,...letter_n
            flab.write('%d,%d,%s\n' % (idx, np.shape(labels)[0], labels_str))


def main():
    convert_ocr_data()

if __name__ == '__main__':
    main()
