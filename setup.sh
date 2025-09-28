# TODO: move to /workspace
python3 -m venv tiny && source tiny/bin/activate
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
# cd DeepSpeedExamples/training/pipeline_parallelism
# # Throughput on 1-4 RTX A5000 24GB
# # No Parallelism: samples/sec: 1150
# deepspeed --num_gpus=1 train.py --deepspeed_config=ds_config.json -p 1 --steps=200
# # Data Parallelism: samples/sec: 1900
# deepspeed --num_gpus=2 train.py --deepspeed_config=ds_config.json -p 1 --steps=200
# # Pipeline Parallelism: samples/sec: 1100
# deepspeed --num_gpus=2 train.py --deepspeed_config=ds_config.json -p 2 --steps=200
# # Data & Pipeline Parallelism: samples/sec: 2000
# deepspeed --num_gpus=4 train.py --deepspeed_config=ds_config.json -p 2 --steps=200

wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvzf cifar-10-python.tar.gz

mkdir out/

python /workspace/SparseTransformers/data/cifar.py

mkdir data_dir

mv train.bin data_dir/
mv val.bin data_dir/

# TODO: move wandb logging to beginning of train file so we log full debug info
# wandb init

nvidia-smi topo -m

# execute these in the SparseTransformers source directory
cd SparseTransformers
# Testing on RTXA5000
# No Parallelism: samples/sec: ~55
deepspeed --num_gpus=1 train_pipeline_parallel.py \
    --deepspeed_config=/config/ds_config.json \
    -p 1 \
    --steps=1000
# Data Parallelism: samples/sec: ~105
deepspeed --num_gpus=2 train_pipeline_parallel.py \
    --deepspeed_config=/config/ds_config.json \
    -p 1 \
    --steps=1000
# Pipeline Parallelism: samples/sec: ~55
deepspeed --num_gpus=2 train_pipeline_parallel.py \
    --deepspeed_config=/config/ds_config.json \
    -p 2 \
    --steps=1000
# Data & Pipeline Parallelism: samples/sec: ~105
deepspeed --num_gpus=4 train_pipeline_parallel.py \
    --deepspeed_config=/config/ds_config.json \
    -p 2 \
    --steps=1000

# Bonus: 4xData Parallelism: samples/sec: 200
deepspeed --num_gpus=4 train_pipeline_parallel.py \
    --deepspeed_config=/config/ds_config.json \
    -p 1 \
    --steps=1000

# Testing 128 layer vanilla model on 1xH200 SXM
# No ZeRO, no checkpointing, no other optimizations
# No Parallelism:
# batch_size: 4; samples/sec: 12.2; HBM: 16GB 
# batch_size: 16; samples/sec: 29.1; HBM: 43GB 
# batch_size: 32; samples/sec: 35.2; HBM: 80GB 
# batch_size: 48; samples/sec: 37.6; HBM: 115GB

deepspeed --num_gpus=1 train_pipeline_parallel.py \
    --deepspeed_config=/config/ds_config.json \
    -p 1 \
    --steps=1000

deepspeed --num_gpus=1 train_llama_pipeline.py \
    --deepspeed_config=/config/ds_config.json \
    -p 1 \
    --steps=1000