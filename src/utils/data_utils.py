import numpy as np
import torch 

def normalize(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr