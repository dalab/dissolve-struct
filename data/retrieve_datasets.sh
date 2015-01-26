GEN_DIR="generated"

if [ ! -d "$GEN_DIR" ]; then
	mkdir $GEN_DIR
fi

cd $GEN_DIR

# Adult
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a

# Forest Cover (Binary)
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
bzip2 -d covtype.libsvm.binary.scale.bz2 

# RCV1
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2
bzip2 -d rcv1_train.binary.bz2

# Factorie jar
wget https://github.com/factorie/factorie/releases/download/factorie-1.0/factorie-1.0.jar
mv factorie-1.0.jar ../dissolve-struct-examples/lib/

cd ..
