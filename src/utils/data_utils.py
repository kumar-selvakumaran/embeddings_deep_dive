import numpy as np
import torch 
import uuid
import pickle

def get_string_hash(object_string):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, object_string)).replace("-", "")

def save_object_to_file(dictionary, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(dictionary, file)

def load_object_from_file(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def normalize(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr