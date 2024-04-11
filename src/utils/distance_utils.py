import torch
import numpy as np

"""
This file contains distance functions of the following format:

INPUTS :

i) embedding array 

ii) array of images


OUTPUTS :

i)  [n x 3] numpy array : image / object ids where each row consists of the 3 best matches (worse to better)
ii) [n x 3] numpy array of numpy arrays : images corresponding to the above image ids 
"""

def get_k_nearest(embedding_matrix, embedding_names, k):
    # embedding_names = np.array(list(embedding_dict.keys()))
    # embedding_matrix = np.array(list(embedding_dict.values()))
    # input_masked_crops  = np.array(list(input_dict.values()))

    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    
    normalized_embeddings = embedding_matrix / norms

    cosine_similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    cosine_similarity_matrix -= np.eye(embedding_matrix.shape[0]) * cosine_similarity_matrix.max()

    neighbour_inds = (cosine_similarity_matrix).argsort(axis =  1)[:, -k:]

    # return embedding_names[neighbour_inds], input_masked_crops[neighbour_inds]    
    return neighbour_inds