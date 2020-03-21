import numpy as np


def get_slices(basis, num_epochs, min_anneal=1, max_anneal=1):

    slices = np.empty((basis.shape[0], num_epochs, basis.shape[1]), dtype=np.float32)
    step = 1. / num_epochs
    for i, p in enumerate(basis):
        slices[i] = np.array([p * z for z in np.arange(min_anneal, max_anneal, step)])
    # epochs x means x dimensions
    return slices.transpose((1, 0, 2))
