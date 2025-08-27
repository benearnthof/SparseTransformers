# Generating Long Sequences with Sparse Transformers
Reproducing the paper Generating Long Sequences with Sparse Transformers by Child et al. In Pytorch
https://arxiv.org/abs/1904.10509

Currently this implementation is a prototype, many fixes and adjustments need to be done to accommodate the original paper. 
So far the trimmed down basic implementation achieves 2.87 bits per byte on CIFAR-10 with a 19M parameter vanilla attention transformer after a short test run of 30k samples. 
The paper trained for 120 epochs of 48k images each so right now I'm satisfied with this very slimmed down prototype. 

What still needs to be implemented: 
* Efficient sparse kernels
* Half size kq projections
* half size linear layers
* activation checkpointing to decrease memory consumption and allow for 128 layer deep model
* ddp training
* proper checkpointing
* proper logging
* correct positional encoding
* correct data loading with respect to the sequence end
* visualization of activation maps like in fig. 1 of the paper
