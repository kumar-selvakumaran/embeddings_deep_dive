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



"""
A class that takes in images, 

stores data in the following format:

object_data (NOTE : it only appending is allowed): 
    # ith index in these arrays corresponds to ith row in the embedding matrix 
    masks : np.arary([np.array(), np.array(), ... #total_objects])
    bounding_boxes : np.array(), [#total_objects x 4]
    source_path : np.array(strs)[1x4],
    class_names : np.array(strs)[1x4]

    
embedding_data (NOTE : it only appending is allowed): 
{
    "<embedding_type>_<black/og background>" : np.array() , shape : [num_objects x flattened_embedding_size]
    .
    .
    .
}

segmentor = SAM
detector = grounding DINO

embedder = embedder is a class following a universal template, where an arbitrary model is used to produce embeddings 
given an image and a BB corresponding to the box

usage : 
    i) extracts objects based on a given prompt from each image of an image list, and stores relevant object details,
    ii) runs the embedder for each of the objects, and appends the embedding matrix
"""
class object_embedder:
    def __init__(self,
                 segmentor = None,
                 detector = None,
                 embeddor = None):
        
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
        
            plot_detections(self.input_image,
                            self.detections,
                            self.prompt_classes,
                            in_window = in_window
                            )
            
class custom_detections:
    def __init__(self,
                 xyxy,
                 masks,
                 confidences,
                 class_ids):
        
        if not(len(boxes) == len(masks) == len(confidences) == len(class_ids)):
            print(f"\nthere must be same # of boxes ({len(boxes)}), masks ({len(masks)}), confs ({len(confidences)}), and cls_ids ({len(class_ids)}). ")
            return -1
        self.xyxy = xyxy
        self.mask = masks
        self.confidence = confidences
        self.class_id = class_ids
            
def enhance_class_name(class_names: List[str]) -> List[str]:
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


