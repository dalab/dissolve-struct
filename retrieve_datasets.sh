cd data/generated

# Adult
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a

# Forest Cover (Binary)
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
bzip2 -d covtype.libsvm.binary.scale.bz2 

# RCV1
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
bzip2 -d rcv1_train.binary.bz2

cd ../..