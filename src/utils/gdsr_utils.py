"""
author = "Kumar Selvakumaran", "Mrudula Acharya", "Neel Adke"
date = "04/24/2024"

Module Docstring
This file contains the required functions and classes load and manage the grounding dino detector SAM segmentor, and Resnet model embeddor
"""

import numpy as np
from typing import List
import cv2
import torch
import os
from IPython.display import Image as im
from IPython.display import display as dis

from torchvision.ops import nms

from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamPredictor

from groundingdino.util.inference import Model

from transformers import AutoImageProcessor, ResNetForImageClassification

from .viz_utils import im_in_window
from .data_utils import get_string_hash, load_object_from_file, save_object_to_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUNDING_DINO_CONFIG_PATH =  "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("../bin/model_files", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join("../bin/model_files", "sam_vit_h_4b8939.pth")


class sam_segmentor:
    """
    A class for image segmentation using a specific segmentation model referred to as SAM.

    Methods:
    __init__(self): Initializes the segmentor with the SAM model.
    segment(image, xyxy): Segments the specified areas in an image based on bounding box coordinates.
    """
    def __init__(self):
        """
        Initializes the segmentor with the SAM model.
        """
        self.segmentor_model = get_sam_model()

    def segment(self,
                image, 
                xyxy):
        """
        Segments the image based on provided bounding boxes using the SAM model.

        Parameters:
            image (numpy array): The image to segment.
            xyxy (list of tuples): List of bounding boxes in the format (xmin, ymin, xmax, ymax).

        Returns:
            numpy array: Array of segmentation masks for the specified bounding boxes.
        """
        self.segmentor_model.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.segmentor_model.predict(
                box=box,
                multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)

        
class resnet_embedder:
    """
    A class to handle embedding extraction using a ResNet model for given detections on images.

    Attributes:
        embeddor_model: A ResNet model loaded and configured for extracting embeddings.

    Methods:
        __init__: Initializes the ResNet model for embedding extraction.
        add_new_embeddings: Processes images and detections to extract and aggregate embeddings.
    """
    def __init__(self):
        """
        Initializes the resnet_embedder with a ResNet model loaded from a predefined function.
        """
        self.embeddor_model = get_resnet_model()
    
    def add_new_embeddings(self,
                           embedding_matrix,
                           image,
                           detections,
                           masks_available = True):
        """
        Adds new embeddings to an existing matrix for detected objects in an image.

        Parameters:
            embedding_matrix (numpy array or None): The current embedding matrix to which new embeddings will be added.
            image (numpy array): The image from which embeddings are extracted.
            detections (object): An object containing detection data including masks and bounding box coordinates.
            masks_available (bool, default=True): Flag indicating whether segmentation masks are available and should be used.

        Returns:
            numpy array: The updated embedding matrix containing new embeddings for the detections.

        Note:
            This method iterates over each detection, optionally uses segmentation masks to focus on detected
            regions, and extracts embeddings using the ResNet model.
        """
        
        num_objects = len(detections)

        print(f"num detections : {num_objects}\n")

        for i in range(num_objects):
            print(f"masks available {masks_available}")
            if masks_available:
                masked_image = image.copy() * np.concatenate([detections.mask[i][..., None],
                            detections.mask[i][..., None],
                            detections.mask[i][..., None]], axis = 2)
            else:
                print(f"skipped_segmentation")
                masked_image = image.copy()
            ymin, xmin, ymax, xmax = detections.xyxy[i].astype(int)
            cropped_image = masked_image[xmin:xmax, ymin:ymax, :]

            cv2.imwrite("temp.jpg", cv2.resize(cropped_image, (300,300)))
            dis(im("temp.jpg"))

            embedding = self.embeddor_model.predict(cropped_image, return_embedding = True).detach().cpu().numpy()
            
            if embedding_matrix is None:
                embedding_matrix = embedding[None, ...]
            else:
                embedding_matrix = np.concatenate([embedding_matrix, embedding[None, ...]])
            
        return embedding_matrix

                
class resnet_model_wrapper:
    """
    A wrapper class for the ResNet model from the Hugging Face Transformers library, designed to process images
    and perform image classification or return embeddings.

    Attributes:
        processor (AutoImageProcessor): The image processor from Hugging Face, tailored for "microsoft/resnet-50".
        model (ResNetForImageClassification): The ResNet model pre-trained on "microsoft/resnet-50".
        device (torch.device): The computation device (CUDA or CPU) where the model will run.

    Methods:
        __init__: Initializes the processor, model, and device settings.
        predict: Processes an image and predicts either the class label or returns an embedding.
    """
    def __init__(self):    
        """
        Initializes the resnet_model_wrapper class by loading the ResNet model and its associated processor
        with pretrained weights. The model is moved to an appropriate device (GPU or CPU).
        """
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)

    def predict(self, image, return_embedding = True):
        """
        Predicts using the ResNet model based on the input image.

        Parameters:
            image (PIL Image or ndarray): The input image to process and predict.
            return_embedding (bool, default=True): Determines if the function should return an embedding vector
                or the class label.

        Returns:
            ndarray or str: Returns an embedding if `return_embedding` is True, otherwise returns the class label as a string.

        Note:
            This function handles image processing, model forwarding, and converts logits to labels or extracts embeddings
            depending on the `return_embedding` parameter.
        """
        inputs = self.processor(image, return_tensors="pt")
        inputs.to(self.device)
        with torch.no_grad():
            if not return_embedding:
                logits = self.model(**inputs).logits
                predicted_label = logits.argmax(-1).item()
                label_string = self.model.config.id2label[predicted_label]
                return label_string
            else:
                embedding = self.model.resnet(**inputs)[1]
                embedding = self.model.classifier[0](embedding)[0]
                return embedding    


def check_model_paths():
    """
    Checks the existence of specified model file paths and prints a message indicating whether each path exists.

    This function iterates over a predefined list of paths, checking if each file path points to an existing file on the disk.
    If a file does not exist, it prints a warning message.
    """
    for imp_file in [GROUNDING_DINO_CHECKPOINT_PATH, GROUNDING_DINO_CONFIG_PATH, SAM_CHECKPOINT_PATH]:
        if not os.path.isfile(imp_file):
            print("\n WARNING !!!!!! : " , GROUNDING_DINO_CONFIG_PATH, " :  DOSENT EXIST:")
        else:
            print(imp_file, " exists")

def get_sam_model(encoder_name = "vit_h"):
    """
    Initializes and returns a SAM model predictor based on a specified encoder.

    Parameters:
        encoder_name (str, optional): The name of the encoder to use within the SAM model. Defaults to "vit_h".

    Returns:
        SamPredictor: An initialized SAM model predictor, configured for the specified encoder and loaded onto the appropriate computational device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAM_ENCODER_VERSION = encoder_name
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

def get_gdino_model():
    """
    Initializes and returns a Grounded DINO model.

    Returns:
        Model: The Grounded DINO model loaded with the specified configuration and checkpoint, ready for inference.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device= device)
    return grounding_dino_model

def get_resnet_model():
    """
    Creates and returns an instance of a ResNet model wrapper.

    Returns:
        resnet_model_wrapper: An instance of the ResNet model wrapper class.
    """
    model = resnet_model_wrapper()
    return model
