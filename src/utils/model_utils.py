"""
author = "Kumar Selvakumaran", "Mrudula Acharya", "Neel Adke"
date = "04/24/2024"

Module Docstring
This class contains the interface functions and classes that uses the embeddor, segmentor and detector , to process data and extract embeddings.
"""
import numpy as np
from typing import List
import cv2
import supervision as sv
import torch
import os

from torchvision.ops import nms

from .viz_utils import im_in_window
from .data_utils import get_string_hash, save_object_to_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUNDING_DINO_CONFIG_PATH =  "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("../bin/model_files", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join("../bin/model_files", "sam_vit_h_4b8939.pth")


class object_embedder:
    """
    A class designed to handle object detection, segmentation, and embedding extraction.

    Attributes:
        segmentor (object): An instance of a segmentation model.
        detector (object): An instance of a detection model.
        embeddor (object): An instance responsible for generating embeddings from detected objects.
        masks (list): List to store masks for detected objects.
        bboxes_xyxy (list): List to store bounding box coordinates for detected objects.
        source_paths (list): List of source image paths processed.
        class_names (list): List of class names for detected objects.
        confidences (list): List of confidence scores for detected objects.
        embedding_matrix (numpy.ndarray): Matrix storing the extracted embeddings for all detected objects.
    """

    def __init__(self,
                 segmentor = None,
                 detector = None,
                 embeddor = None):
        """
        Initializes the object_embedder with provided models.

        Parameters:
            segmentor (object, optional): The segmentation model.
            detector (object, optional): The detection model.
            embeddor (object, optional): The embedding model.
        """
        
        self.segmentor, self.detector, self.embeddor = segmentor, detector, embeddor

        self.masks = []
        self.bboxes_xyxy = []
        self.source_paths = []
        self.class_names = []
        self.confidences = []
        self.embedding_matrix = None

    def detect_objects(self,
                        image_paths,
                        classes,
                        box_threshold = 0.40,
                        text_threshold = 0.25,
                        do_segment = True,
                        viz_outputs = False,
                        in_window = False,
                        model_type = 'unknown'):
        """
        Processes a list of image paths to detect objects, perform segmentation, and extract embeddings.

        Parameters:
            image_paths (list): List of paths to the images.
            classes (list): List of class names.
            box_threshold (float): Confidence threshold for bounding box detection.
            text_threshold (float): Confidence threshold for text detection.
            do_segment (bool): Flag to perform segmentation on detected objects.
            viz_outputs (bool): Flag to visualize the detection results.
            in_window (bool): Flag to display results in a separate window.
            model_type (str): Type of model used for processing, used for saving data.

        Outputs:
            Processes each image, detecting objects, optionally segmenting them, extracting embeddings, and potentially visualizing the results.
        """
        
        classes.append('No Class')

        print(f"\nclasses : {classes[:-1]}\n")

        with torch.no_grad():
            for image_path in image_paths:
                print(f"\nimage path : {image_path}\n")
                image = cv2.imread(image_path)

                """
                detections class as done by GDINO should be made by future models as well
                """
                detections = self.detector.predict_with_classes(
                    image=image,
                    classes=enhance_class_name(class_names=classes[:-1]),
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )

                """
                if self.segmentor is None, (which we pass if we dont want segmentation), then the segment functions should 
                return masks corresponding to the bounding boxes.

                the segment function will be unique to each segementor, whether it is sam or other models. so let 'segment'be a member function
                of the self.segmentor object
                """
                if do_segment:
                    detections.mask = self.segmentor.segment(
                        # sam_predictor=self.segmentor,
                        image=cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB),
                        xyxy=detections.xyxy
                    )   
                

                boxes = torch.tensor(detections.xyxy)
                scores = torch.tensor(detections.confidence)

                # inter-class NMS, to wrong class Bounding Box duplicates
                keep_inds = nms(boxes, scores, 0.99).cpu().numpy()

                detections.xyxy = detections.xyxy[keep_inds]                
                detections.confidence = detections.confidence[keep_inds]
                detections.class_id = detections.class_id[keep_inds]

                if do_segment:
                    detections.mask = detections.mask[keep_inds]
                
                class_ids = detections.class_id
                class_ids[class_ids == None] = len(classes) - 1
                class_ids = class_ids.astype(int)
                
                classes = np.array(classes)
                num_objects = len(detections)

                if num_objects == 0:
                    continue

                if do_segment:
                    self.masks += np.split(detections.mask, num_objects)
                
                self.bboxes_xyxy += np.split(detections.xyxy, num_objects)
                self.class_names += classes[class_ids].tolist()
                self.confidences += detections.confidence.tolist()
                self.source_paths += [image_path for i in range(num_objects)]

                result_str = "\n".join([f"{classes[class_id]} : {len(class_ids[class_ids == class_id])}" for class_id in np.unique(class_ids)])
                print(f"results : \n {result_str}")

                print(f"")
                self.embedding_matrix = self.embeddor.add_new_embeddings(self.embedding_matrix,
                                                                         image.copy(),
                                                                         detections,
                                                                         masks_available = do_segment)

                print(f"embedding_matrix shape : {self.embedding_matrix.shape}")

                if viz_outputs == True:
                    plot_detections(image.copy(),
                                    detections,
                                    classes,
                                    in_window = in_window
                                    )
        
        self.save_data(model_type = model_type)
        
    def save_data(self,
                  model_type = 'unknown'):
        """
        Saves detection and embedding data to files.

        Parameters:
            model_type (str): Descriptive string for the type of model, appended to save filenames.

        Outputs:
            Saves masks, bounding boxes, confidences, class names, source paths, and embeddings to respective files.
        """
        
        dataset_hash_key = " ".join(self.source_paths) 
        dataset_hash_name = get_string_hash(dataset_hash_key)  + f"_{model_type}"   
        
        save_path_details = os.path.join("/app/bin/results/" , f'embedding_details_{dataset_hash_name}.pkl')
        save_path_matrix = os.path.join("/app/bin/results/" , f'embedding_matrix_{dataset_hash_name}.pkl')
        
        data_object = {
            "masks" : self.masks,
            "bounding_boxes" : self.bboxes_xyxy,
            "confidences" : self.confidences,
            "class_names" : self.class_names,
            "source_paths" : self.source_paths,
            "embedding_matrix_path" : save_path_matrix
        }

        print(f"\nsaved file name : {dataset_hash_name}\n")
        save_object_to_file(data_object, save_path_details)
        save_object_to_file(self.embedding_matrix, save_path_matrix)


    def plot_results(self,
                     in_window = False):
        """
        Visualizes the detection results.

        Parameters:
            in_window (bool): Flag to display results in a separate window.
        """
    
        plot_detections(self.input_image,
                        self.detections,
                        self.prompt_classes,
                        in_window = in_window
                        )
            
def enhance_class_name(class_names: List[str]) -> List[str]:
    """formats a list of class names"""
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


def plot_detections(image,
                    detections,
                    classes,
                    in_window = False,
                    do_segment = True
                    ):
    """
    Annotates and displays an image with detection results including bounding boxes and segmentation masks.

    Parameters:
        image (numpy array): The image to annotate and display.
        detections (list): List of detection tuples with coordinates, confidence, and class indices.
        classes (list of str): Class names corresponding to the class indices in detections.
        in_window (bool, optional): If True, displays the image in a separate window; otherwise inline.
        do_segment (bool, optional): If True, applies segmentation masks to the detections.

    Uses `BoxAnnotator` and `MaskAnnotator` for drawing bounding boxes and segmentation masks respectively.
    """
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    if do_segment:
        image = mask_annotator.annotate(scene=image, detections=detections)

    image = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    if not in_window:
        sv.plot_image(image, (16, 16))
    
    else:
        im_in_window(image, "detections")


