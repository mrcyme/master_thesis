import sobol_seq
import pandas as pd
import numpy as np
import pyDOE



def sobol_sampling(config,N):
    # Get the range for the support
    ranges = [interval['max_value'] - interval['min_value'] for interval in config.values()]
    inc = [interval['min_value'] for interval in config.values()]
    vec = sobol_seq.i4_sobol_generate(len(config.keys()), N)
    sobol_samples = np.vstack([np.multiply(v, ranges) + inc for v in vec])
    return sobol_samples


def lhs_sampling(config,N):
    ranges = [interval['max_value'] - interval['min_value'] for interval in config.values()]
    inc = [interval['min_value'] for interval in config.values()]
    vec= pyDOE.lhs(len(config.keys()), samples=N)
    lhs_samples= np.vstack([np.multiply(v, ranges) + inc for v in vec])
    return lhs_samples


def random_sampling(config,N):
    ranges = [interval['max_value'] - interval['min_value'] for interval in config.values()]
    inc = [interval['min_value'] for interval in config.values()]
    vec = np.random.rand(len(config),N).T
    random_samples= np.vstack([np.multiply(v, ranges) + inc for v in vec])
    return random_samples
