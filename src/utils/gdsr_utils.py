
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
    def __init__(self):
        self.segmentor_model = get_sam_model()

    def segment(self,
                image, 
                xyxy):
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
    def __init__(self):
        self.embeddor_model = get_resnet_model()
    
    def add_new_embeddings(self,
                           embedding_matrix,
                           image,
                           detections,
                           masks_available = True):
        
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
    def __init__(self):    
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)

    def predict(self, image, return_embedding = True):
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
    
    for imp_file in [GROUNDING_DINO_CHECKPOINT_PATH, GROUNDING_DINO_CONFIG_PATH, SAM_CHECKPOINT_PATH]:
        if not os.path.isfile(imp_file):
            print("\n WARNING !!!!!! : " , GROUNDING_DINO_CONFIG_PATH, " :  DOSENT EXIST:")
        else:
            print(imp_file, " exists")

def get_sam_model(encoder_name = "vit_h"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SAM_ENCODER_VERSION = encoder_name
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=device)
    sam_predictor = SamPredictor(sam)
    return sam_predictor

def get_gdino_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH, device= device)
    return grounding_dino_model

def get_resnet_model():
    model = resnet_model_wrapper()
    return model

# def segment_sam(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
#     sam_predictor.set_image(image)
#     result_masks = []
#     for box in xyxy:
#         masks, scores, logits = sam_predictor.predict(
#             box=box,
#             multimask_output=True
#         )
#         index = np.argmax(scores)
#         result_masks.append(masks[index])
#     return np.array(result_masks)

