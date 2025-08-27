wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz

mkdir out/

mv train.bin data_dir/
mv val.bin data_dir/
