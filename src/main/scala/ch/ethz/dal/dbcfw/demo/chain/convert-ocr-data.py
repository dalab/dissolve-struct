import numpy as np
import scipy.io
import sys

def main():

    if len(sys.argv) != 2:
        sys.exit("Format: convert-ocr-data.py ocr.mat")

    mat = scipy.io.loadmat(sys.argv[1], struct_as_record=False, squeeze_me=True)
    n = np.shape(mat['dataset'])[0]
    with open('patterns.csv', 'w') as fpat, open('labels.csv', 'w') as flab, open('folds.csv', 'w') as ffold:
        for i in range(n):
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
            fpat.write('%d,%d,%d,%s\n' % (i+1, np.shape(xi)[0], np.shape(xi)[1], xi_str))

            ### Write labels (y_i's)
            labels = mat['dataset'][i].__dict__['word']
            labels_str = ','.join([`a` for a in labels])
            # FORMAT: id,#letters,letter_0,letter_1,...letter_n
            flab.write('%d,%d,%s\n' % (i+1, np.shape(labels)[0], labels_str))

            ### Write folds
            fold = mat['dataset'][i].__dict__['fold']
            # FORMAT: id,fold
            ffold.write('%d,%d\n' % (i+1, fold))

if __name__ == '__main__':
    main()
