import pickle as cPickle
import numpy as np
import random
from pathlib import Path
import os

def load_train():
    path = Path('./cifar-10-batches-py')
    filestring = 'data_batch_'
    contents = [path / x for x in os.listdir(path) if filestring in x]

    images = [cPickle.load(open(f, "rb"), encoding="latin1") for f in contents]

    images = [x["data"] for x in images]
    imagearray = np.array(images)   #   (5, 10000, 3072)
    return np.vstack(imagearray) # (50000, 3072)

def load_test():
    path = Path('./cifar-10-batches-py')
    filestring = 'test_batch'
    contents = [path / x for x in os.listdir(path) if filestring in x]

    images = [cPickle.load(open(f, "rb"), encoding="latin1") for f in contents]

    images = [x["data"] for x in images]
    imagearray = np.array(images)   #   (5, 10000, 3072)
    return np.vstack(imagearray) # (50000, 3072)

train_data = load_train()
val_data = load_test()
# data is already encoded as raw bytes
# export to bin files

train_data.tofile('train.bin')
val_data.tofile('val.bin')
