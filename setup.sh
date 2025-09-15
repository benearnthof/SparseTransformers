git clone https://github.com/benearnthof/SparseTransformers.git

wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz

mkdir out/

python /root/SparseTransformers/data/cifar.py

mkdir data_dir

mv train.bin data_dir/
mv val.bin data_dir/

pip install wandb
pip install omegaconf
pip install matplotlib
pip install deepspeed
pip install mpi4py

apt-get update
apt-get install -y libopenmpi-dev openmpi-bin
pip install --force-reinstall --no-cache-dir mpi4py
# make sure to export these on new GPU nodes
export DEEPSPEED_COMM_BACKEND=nccl
export NCCL_P2P_LEVEL=SYS
# for multinode? 

nvidia-smi topo -m
wandb init

torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py

