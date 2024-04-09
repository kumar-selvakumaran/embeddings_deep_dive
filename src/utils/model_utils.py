
import numpy as np
from typing import List
import cv2
import supervision as sv
import torch
import os

from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamPredictor

from groundingdino.util.inference import Model

from transformers import AutoImageProcessor, ResNetForImageClassification

from .viz_utils import im_in_window

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GROUNDING_DINO_CONFIG_PATH =  "../GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("../bin/model_files", "groundingdino_swint_ogc.pth")
SAM_CHECKPOINT_PATH = os.path.join("../bin/model_files", "sam_vit_h_4b8939.pth")


class object_isolator:
    def __init__(self):
        self.segmentor = get_sam_model("vit_h")
        self.detector = get_gdino_model()
        self.object_dict = {}
        self.available_classes = {}
        self.prompt = ""
        self.embedder = get_resnet_model()

    def detect_objects(self,
                        image,
                        classes,
                        box_threshold,
                        text_threshold,
                        viz_outputs = False):
        
                
        detections = self.detector.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=classes),
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        
        detections.mask = segment(
            sam_predictor=self.segmentor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )   

        class_ids  =np.array(detections.class_id)
        class_groups = [np.nonzero(class_ids == class_id)[0].tolist() for class_id in np.unique(class_ids)]
        object_dict = dict(zip(np.unique(class_ids), class_groups))
        self.object_dict = object_dict

        result_str = "\n".join([f"{classes[class_id]} : {len(class_groups)}" for class_id, class_groups in object_dict.items()])
        print(f"results : \n {result_str}")


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

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]


def plot_detections(image,
                    detections,
                    classes,
                    in_window = False
                    ):
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{classes[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, _
        in detections]
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    if not in_window:
        sv.plot_image(annotated_image, (16, 16))
    
    else:
        im_in_window(annotated_image, "detections")

