"""
author = "Kumar Selvakumaran", "Mrudula Acharya", "Neel Adke"
date = "04/24/2024"

Module Docstring
The contents of this file is used to store and load embeddings and their relevant details
"""

import numpy as np
import torch 
import uuid
import pickle

def get_string_hash(object_string):
    """
    Generate a unique hash for a given string using UUID5.

    Parameters:
    object_string (str): The string to hash.

    Returns:
    str: A unique hash string derived from the input string.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, object_string)).replace("-", "")

def save_object_to_file(dictionary, file_path):
    """
    Save a Python dictionary object to a file using pickle.

    Parameters:
    dictionary (dict): The dictionary to save.
    file_path (str): The file path where the dictionary should be saved.
    """
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)

def load_object_from_file(file_path):
    """
    Load a Python dictionary object from a file using pickle.

    Parameters:
    file_path (str): The file path from which to load the dictionary.

    Returns:
    dict: The dictionary loaded from the file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def normalize(arr):    
    """
    Normalize an array to the range [0, 1].

    Parameters:
    arr (np.array): The numpy array to normalize.

    Returns:
    np.array: The normalized array where all values are scaled to the range [0, 1].
    """
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr