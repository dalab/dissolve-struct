import numpy as np
import scipy.io
import sys

FILTER_FOLD = True
TST_FOLD_NUMS = [0,] # Datapoints from these folds will be treated as test dataset

def main():

    if len(sys.argv) != 2:
        sys.exit("Format: convert-ocr-data.py ocr.mat")

    idx_trn = 0
    idx_tst = 0

    mat = scipy.io.loadmat(sys.argv[1], struct_as_record=False, squeeze_me=True)
    n = np.shape(mat['dataset'])[0]
    with open('patterns_train.csv', 'w') as fpat_trn, open('labels_train.csv', 'w') as flab_trn, open('folds_train.csv', 'w') as ffold_trn, \
            open('patterns_test.csv', 'w') as fpat_tst, open('labels_test.csv', 'w') as flab_tst, open('folds_test.csv', 'w') as ffold_tst:
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

if __name__ == '__main__':
    main()
