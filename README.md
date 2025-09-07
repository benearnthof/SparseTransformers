# Generating Long Sequences with Sparse Transformers
Reproducing the paper Generating Long Sequences with Sparse Transformers by Child et al. In Pytorch
https://arxiv.org/abs/1904.10509

Currently this implementation is a prototype, some adjustments need to be made to accommodate the original paper. 
The 128 layer, 2 head dense attention baseline can successfully overfit on a subset of CIFAR-10 to zero loss:
<img src="https://github.com/benearnthof/SparseTransformers/blob/main/assets/overfit_cifar.gif" alt="https://github.com/benearnthof/SparseTransformers/blob/main/assets/overfit_cifar.gif" width="600"/>  
The paper trained for 120 epochs of 48k images each (5760000 samples total) so right now I'm satisfied with this very slimmed down prototype. 

## Notes and todo list  
### Memory profiling & Performance
#### Features
* Activation Checkpointing to decrease GPU memory requirements for very deep transformers
* Automatic mixed precision
* DDP Training via `torchrun --standalone --nproc_per_node=2 train.py config/cifar-10-ddp.yaml`
* During evaluation masked images are sampled and logged on wandb with their respective predictions
* To avoid data leakage from image to image the last target byte y[-1] is set to the last input byte x[-1]  
* Adjusted positional encodings for image data, see below.
* GPU memory profiling can be enabled with `debug_memory` in config.
* To visualize the .pickle files this produces head over to [https://docs.pytorch.org/memory_viz](https://docs.pytorch.org/memory_viz)

#### TODO
* Compare Vanilla to FlashAttention on different hardware  
* Examine impact of batch size on training, as larger batch sizes may be beneficial for transformers, but gradient accumulation & activation checkpointing do have small performance drawbacks.  
* Investigate NCCL_P2P_DISABLE=1 / export NCCL_P2P_LEVEL=NVL may be required for some GPUs

### Functionality
* Parameters & Embeddings are initialized like specified in section 6 of the paper
* Dropout is only applied at the end of each residual addition as per section 5.4 of the paper  
* Pre-activation residual block of https://arxiv.org/pdf/1603.05027  
* Feed-forward networks project to mlp_dim * d, with mlp_dim 4 as standard, mlp_dim 2 for "half-size" as outlined in section 7.1 for CIFAR-10  
* Layer-dependent weight initialization
```python
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

#### TODO
* resume training from checkpoint  
* Visualize positional encodings to clarify what's going on  
* Try training with sparse pytorch implementation & torch.compile  
* Implement blocksparse CUDA kernels  
* add other masking variants (bidirectional training should only require adding a couple lines of code)  
* Data augmentation à la https://arxiv.org/abs/1909.13719  
* Learning rate scaling with batch size https://arxiv.org/pdf/1706.02677  
* Stochastic Depth https://arxiv.org/pdf/1603.09382  
* Weight init with truncated Gaussian https://arxiv.org/pdf/1803.01719  
* Try the GPT-2 style architecture from https://proceedings.mlr.press/v119/chen20s/chen20s.pdf  

### Visualization
#### TODO
* attention visualization for masked images
* visualize attention matrices for checkpoint (maybe during training?) 
* visualize categorial output distribution during training 

### Additional Resources
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html  
https://pytorch.org/blog/activation-checkpointing-techniques/  
https://docs.pytorch.org/docs/stable/checkpoint.html  

### On positional Encoding
In addition to the embedding of input symbols, positional embeddings are typically used in Transformers and other location-agnostic architectures to encode the spatial relationships of data (Gehring et al., 2017), (Parmar et al., 2018).

We found using learned embeddings which either encoded the structure of the data or the factorized attention patterns were important for performance of our models.

We added either $n_{e m b}=d_{d a t a}$ or $n_{e m b}=d_{a t t n}$ embeddings to each input location, where $d_{\text {data }}$ refers to the number of dimensions of the data, and $d_{a t t n}$ is the number of dimensions of the factorized attention. If $\mathbf{x}_i$ is the one-hot encoded $i$ th element in the sequence, and $\mathbf{o}_i^{(j)}$ represents the one-hot encoded position of $x_i$ in the $j$-th dimension $\left(1 \leq j \leq n_{e m b}\right)$, then:

```math
embed\left(X, W_e\right)=\left(\mathbf{x}_i W_e+\sum_{j=1}^{n_{e m b}} \mathbf{o}_i^{(j)} W_j\right)_{\mathbf{x}_i \in X}
```

For images, we used data embeddings, where $d_{\text {data }}=3$ for the row, column, and channel location of each input byte. For text and audio, we used two-dimensional attention embeddings, where $d_{\text {attn }}=2$ and the index corresponds to each position's row and column index in a matrix of width equal to the stride.

Implementing this should look something like this: 
```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # positional encoding for CIFAR-10 images
            row_emb = nn.Embedding(32, config.n_embd),  # 32 rows
            col_emb = nn.Embedding(32, config.n_embd),  # 32 cols
            chan_emb = nn.Embedding(3, config.n_embd),  # 3 channels (RGB)
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        ...

def forward(self, idx, targets=None):
    device = idx.device
    b, t = idx.size()

    tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
    # positional embedding for CIFAR-10 Images
    # decode flat indices (0..3071) into row, col, chan
    H, W, C = 32, 32, 3
    # we load CIFAR-10 images as 3072 bytes where the first 1024 correspond to the first 
    # channel of the image, 32 bytes of which correspond to the first row.
    positions = torch.arange(t, device=device)
    chans = positions // (H * W)                 # 0..2
    rows  = (positions % (H * W)) // W           # 0..31
    cols  = positions % W              
    row_emb = self.transformer.row_emb(rows)[None, :, :].expand(b, -1, -1)
    col_emb = self.transformer.col_emb(cols)[None, :, :].expand(b, -1, -1)
    chan_emb = self.transformer.chan_emb(chans)[None, :, :].expand(b, -1, -1)

    x = self.transformer.drop(tok_emb + row_emb + col_emb + chan_emb)
    ...
    
```
Where we can in turn remove the old weighted position embedding.  

### On Embeddnig Initialization
All embeddings are of a constant dimension $d$, usually one of $\{256,512,1024\}$. By default, all linear transforms are to the same dimension, with the exception of the feed-forward network, which projects the input to $4 d$, unless we use "half-size" transformations, where it is $2 d$. Additionally, sometimes we halve the size of the query and key transformations.

We initialize the token embedding $W_e$ from $\mathcal{N}\left(0, \frac{0.125}{\sqrt{d}}\right)$ and the position embeddings from $\mathcal{N}\left(0, \frac{0.125}{\sqrt{d n_{e m b}}}\right)$. Within the attention and feedforward components, all biases are initialized to 0 and all weights are initialized from $\mathcal{N}\left(0, \frac{0.125}{\sqrt{d_{i n}}}\right)$ where $d_{i n}$ is the fan-in dimension. The weight matrix for the output logits was initialized to 0.

## Training & GPU Considerations
### Overfitting the Baseline
cifar-10-overfit.yaml produces a 7,488,256 parameter model, using about 7GB of GPU HBM.  
Dense Attention (flashattention) with full training config uses ~58GB on a A100 80GB at batch size 16 for CIFAR-10.  

For activation checkpointing at batch size 16 peak memory usage is as follows:  

| Remat Steps | Peak HBM |
|-------------|----------|
| 1           | 58 GB    |
| 2           | 33 GB    |
| 4           | 19 GB    |
| 8           | 13 GB    |  

Activation checkpointing achieves the following throughput  

| Remat Steps | Max Batchsize | Peak HBM | Time/Batch | Time/Epoch |
|-------------|------------|----------|------------|------------|
| 1           | 16         | 58GB     | 910ms      | 0.79h      |
| 2           | 32         | 59GB     | 1850ms     | 0.80h      |
| 4           | 64         | 60GB     | 3780ms     | 0.82h      |
| 8           | 128        | 63GB     | 7780ms     | 0.84h      |  

The optimizer state seems to take up around 5 GB.  

This brings the total training time (without early stopping) on an 8xA100 80GB node to about 18 hours and 45 minutes, which, as of September 2025, yields a cost of around $250 depending on the cloud GPU provider. The number of activation checkpoints can be set in the model config.  
I will do some more benchmarks to look at the performance per dollar of various other GPUs, low-demand options like the RTX A6000 48GB could be a sensible middle ground.  
