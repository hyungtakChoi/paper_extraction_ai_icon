import torch
import random
import numpy as np
import multiprocessing as mp

INF_DATASET_SIZE = 8

def set_seed(seed):
    """Set the global seed for the whole works to be executed."""
    random.seed(seed) # 1. Python built-in RNG
    np.random.seed(seed) # 2. Numpy RNG
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # 3. Torch CUDA RNG; apply to all cuda devices
    torch.manual_seed(seed) # 4. Torch RNG

def set_multiprocessing():
    """Set up for multiprocessing.

    This function should be called before processing Dataset via multiple
    processes, for example, when you specify `num_proc`>=1 in
    Dataset.map(num_proc).
    """
    mp.set_start_method("spawn", force=True)

