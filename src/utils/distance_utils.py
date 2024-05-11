"""
author = "Kumar Selvakumaran", "Mrudula Acharya", "Neel Adke"
date = "04/24/2024"

Module Docstring
This file contains the required function to find K nearest neighbors
"""

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
    """
    Computes the k-nearest neighbors for each embedding in the matrix using cosine similarity.

    Parameters:
        embedding_matrix (numpy.ndarray): The matrix of embeddings where each row is an embedding.
        embedding_names (numpy.ndarray): Array of names corresponding to the embeddings.
        k (int): The number of nearest neighbors to retrieve for each embedding.

    Returns:
        numpy.ndarray: An array of indices corresponding to the k-nearest neighbors for each embedding.

    Note:
        The function normalizes the embeddings, computes the cosine similarity matrix, and retrieves the indices
        of the top k nearest neighbors for each embedding, excluding the self-similarity.
    """

    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    
    normalized_embeddings = embedding_matrix / norms

    cosine_similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
    
    cosine_similarity_matrix -= np.eye(embedding_matrix.shape[0]) * cosine_similarity_matrix.max()

    neighbour_inds = (cosine_similarity_matrix).argsort(axis =  1)[:, -k:]
  
    return neighbour_inds