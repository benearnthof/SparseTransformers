# Notes and todolist

## Memory profiling & Performance  
https://pytorch.org/blog/activation-checkpointing-techniques/  
https://docs.pytorch.org/docs/stable/checkpoint.html  
With 8 checkpointing splits memory usage is at 18% (!) for batch size 16. Batch size 64 uses 50GB memory but wall time per epoch is the same for 32 and 64. (around 50 minutes).
* Flame Graph profiling of model to visualize memory usage before and after gradient checkpointing
* Move number of checkpointing steps to config
* Dense Attention (flashattention) with full training config uses 80% of A100 memory at 
batch size 16 for cifar-10
* Compare Vanilla to FlashAttention on different hardware
* DDP Training
### Automatic mixed precision/Mixed precision training: 
https://docs.pytorch.org/docs/stable/amp.html
already implemented with ctx context manager

## Functionality
* sample during training & eval, log images on wandb
* Adjust the positional encoding for image data
* Implement sparse kernels
* Apply Dropout only at the end of each residual addition.
* Use pre-activation residual block of https://arxiv.org/pdf/1603.05027
* Fix last byte leak in data loader (We should repeat the penultimate pixel twice to avoid cross image contamination)
* resume from checkpoint

## Visualization
* attention visualization for masked images
* visualize attention matrices for checkpoint (maybe during training?) 

current best with flawed implementation: 
iter 610: loss 3.3404, time 8406.33ms, mfu 57.64%
iter 611: loss 3.3030, time 8354.34ms, mfu 57.65%
iter 612: loss 3.2933, time 8343.33ms, mfu 57.67%
iter 613: loss 3.3958, time 8357.51ms, mfu 57.67%
iter 614: loss 3.3052, time 8345.89ms, mfu 57.69%
iter 615: loss 3.2869, time 8353.94ms, mfu 57.69%
iter 616: loss 3.3050, time 8343.27ms, mfu 57.70%

# Additional Resources
https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial15/Vision_Transformer.html
