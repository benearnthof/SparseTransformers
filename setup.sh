# TODO: move to /workspace
python3 -m venv venv && source venv/bin/activate
python -m pip install --upgrade pip

pip install "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
  --index-url https://download.pytorch.org/whl/cu128

pip install "deepspeed==0.17.6" "transformers==4.44.2" "accelerate==1.0.1" \
  "omegaconf" "matplotlib" "wandb"

# pip install "numpy<2.0.0"

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# minimal example to test pipeline parallelism
# git clone https://github.com/deepspeedai/DeepSpeedExamples.git
cd DeepSpeedExamples/training/pipeline_parallelism
deepspeed --num_gpus=1 train.py --deepspeed_config=ds_config.json -p 1 --steps=200


wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz

mkdir out/

python /root/SparseTransformers/data/cifar.py

mkdir data_dir

mv train.bin data_dir/
mv val.bin data_dir/

# make sure to export these on new GPU nodes
# export DEEPSPEED_COMM_BACKEND=nccl
# export NCCL_DEBUG=INFO
# for A5000 GPUs
# export NCCL_P2P_DISABLE=1
# export NCCL_P2P_LEVEL=NVL
# export NCCL_SOCKET_IFNAME=ens1
# export NCCL_P2P_LEVEL=SYS
# # for multinode? 

# TODO: move wandb logging to beginning of train file so we log full debug info
# wandb init

nvidia-smi topo -m

cd SparseTransformers
deepspeed --num_gpus=1 train_pipeline_parallel.py \
    --deepspeed_config=/config/ds_config.json \
    -p 1 \
    --steps=5000














# pp stages 1 in config
torchrun --nproc_per_node=1 train_pipeline_parallel.py
# pp stages 1 in config => automatic data parallel ? 
torchrun --nproc_per_node=2 train_pipeline_parallel.py


torchrun --nproc_per_node=2 disttest.py


deepspeed --num_gpus=2 train_pipeline_parallel.py --deepspeed_config=ds_config.json --pipeline-parallel-size=2 --steps=10
