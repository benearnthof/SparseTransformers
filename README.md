# Generating Long Sequences with Sparse Transformers
Reproducing the paper Generating Long Sequences with Sparse Transformers by Child et al. In Pytorch
https://arxiv.org/abs/1904.10509

Currently this implementation is a prototype, some adjustments need to be made to accommodate the original paper. 
The baseline implementation can successfully overfit a subset of CIFAR-10 to zero loss:
<img src="https://raw.githubusercontent.com/benearnthof/SparseTransformers/refs/heads/main/assets/overfit.jpg" alt="https://raw.githubusercontent.com/benearnthof/SparseTransformers/refs/heads/main/assets/overfit.jpg" width="400"/>  
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

#### TODO
* Flame Graph profiling of model to visualize memory usage before and after gradient checkpointing & compare impact of hardware
* Compare Vanilla to FlashAttention on different hardware  
* examine impact of batch size on training stability  
* larger batch sizes may be beneficial for transformers  
* investigate NCCL_P2P_DISABLE=1 / export NCCL_P2P_LEVEL=NVL may be required for some GPUs

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
* Implement sparse kernels
* resume training from checkpoint
* add other masking variants (data augmentation ?)
* Visualize positional encodings to clarify what's going on
* Consider tweaks to enhance data efficiency https://arxiv.org/abs/2012.12877

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

We added either $n_{e m b}=d_{d a t a}$ or $n_{e m b}=d_{a t t n}$ embeddings to each input location, where $d_{\text {data }}$ refers to the number of dimensions of the data, and $d_{a t t n}$ is the number of dimensions of the factorized attention. If $\mathbf{x}_i$ is the one-hot encoded $i$ th element in the sequence, and $\mathbf{o}_i^{(j)}$ represents the one-hot encoded position of $\mathrm{x}_i$ in the $j$ th dimension $\left(1 \leq j \leq n_{e m b}\right)$, then:

$$
\operatorname{embed}\left(X, W_e\right)=\left(\mathbf{x}_i W_e+\sum_{j=1}^{n_{e m b}} \mathbf{o}_i^{(j)} W_j\right)_{\mathbf{x}_i \in X}
$$
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
* cifar-10-overfit.yaml produces a 7,488,256 parameter model, using about 7GB of GPU HBM.
* Dense Attention (flashattention) with full training config uses 80% of A100 memory at batch size 16 for CIFAR-10.  
* With 8 checkpointing splits memory usage is at 18% (!) for batch size 16. Batch size 64 uses 50GB memory but wall time per epoch is the same for 32 and 64. (around 50 minutes).
* The number of activation checkpoints can be set in the model config.
* With DDP both nodes of RTX A6000 48GB and A40 48GB seem attractive, still need to benchmark optimal training configurations & compare wall clock time per epoch & implied total training cost.
* RTX A6000 48GB: Batch size 64, Grad Accumulation steps: 4, time per iter: 5800ms, ~0.16 hours per epoch
