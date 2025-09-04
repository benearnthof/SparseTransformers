# Generating Long Sequences with Sparse Transformers
Reproducing the paper Generating Long Sequences with Sparse Transformers by Child et al. In Pytorch
https://arxiv.org/abs/1904.10509

Currently this implementation is a prototype, some adjustments need to be made to accommodate the original paper. 
So far the trimmed down basic implementation can train on CIFAR-10. A 50m parameter dense attention transformer achieved the following results after ~4 epochs:  
<img src="https://raw.githubusercontent.com/benearnthof/SparseTransformers/refs/heads/main/assets/eval_3000.jpg" alt="https://raw.githubusercontent.com/benearnthof/SparseTransformers/refs/heads/main/assets/eval_3000.jpg" width="400"/>  
The paper trained for 120 epochs of 48k images each (5760000 samples total) so right now I'm satisfied with this very slimmed down prototype. 

## Notes and todo list  
### Memory profiling & Performance
https://pytorch.org/blog/activation-checkpointing-techniques/  
https://docs.pytorch.org/docs/stable/checkpoint.html  
With 8 checkpointing splits memory usage is at 18% (!) for batch size 16. Batch size 64 uses 50GB memory but wall time per epoch is the same for 32 and 64. (around 50 minutes).
* Flame Graph profiling of model to visualize memory usage before and after gradient checkpointing
* Move number of checkpointing steps to config
* Dense Attention (flashattention) with full training config uses 80% of A100 memory at 
batch size 16 for cifar-10
* Compare Vanilla to FlashAttention on different hardware
* DDP Training
* examine impact of batch size on training stability
* larger batch sizes may be beneficial for transformers
#### Automatic mixed precision/Mixed precision training: 
https://docs.pytorch.org/docs/stable/amp.html
already implemented with ctx context manager

### Functionality
#### Features
* During evaluation masked images are sampled and logged on wandb with their respective predictions
* To avoid data leakage from image to image the y[-1] is set to x[-1]
#### TODO
* Adjust the positional encoding for image data
* Implement sparse kernels
* Apply Dropout only at the end of each residual addition.
* Use pre-activation residual block of https://arxiv.org/pdf/1603.05027
* resume training from checkpoint

### Visualization
* attention visualization for masked images
* visualize attention matrices for checkpoint (maybe during training?) 

### Additional Resources
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html

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
self.row_emb = nn.Embedding(32, config.n_embd)     # 32 rows
self.col_emb = nn.Embedding(32, config.n_embd)     # 32 cols
self.chan_emb = nn.Embedding(3, config.n_embd)     # RGB channels

tok_emb = self.transformer.wte(idx)  # (B, T, n_embd)

# decode the flattened index (0..3071) into (row, col, channel)
b, t = idx.shape
rows = torch.arange(0, 32, device=idx.device).repeat_interleave(32*3)
cols = torch.arange(0, 32, device=idx.device).repeat(32*3)
chans = torch.arange(0, 3, device=idx.device).repeat(32*32)

rows = rows[:t]
cols = cols[:t]
chans = chans[:t]

row_emb = self.row_emb(rows)[None, :, :].expand(b, -1, -1)
col_emb = self.col_emb(cols)[None, :, :].expand(b, -1, -1)
chan_emb = self.chan_emb(chans)[None, :, :].expand(b, -1, -1)

x = tok_emb + row_emb + col_emb + chan_emb
x = self.transformer.drop(x)
```
Where we can in turn remove the old weighted position embedding.

