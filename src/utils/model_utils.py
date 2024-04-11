
import numpy as np
from typing import List
import cv2
import supervision as sv
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

        if segmentor is None:
            segmentor = get_sam_model("vit_h")
        
        if detector is None:
            self.detector = get_gdino_model()

        self.masks = []
        self.bboxes_xyxy = []
        self.source_paths = []
        self.class_names = []
        self.confidences = []
        self.embedding_matrix = None

        if self.embeddor is None:
            self.embeddor = get_resnet_model()
            
    def detect_objects(self,
                        image_paths,
                        classes,
                        box_threshold = 0.40,
                        text_threshold = 0.25,
                        viz_outputs = False,
                        in_window = False):
        
        classes.append('No Class')

        print(f"\nclasses : {classes[:-1]}\n")

        with torch.no_grad():
            for image_path in image_paths:
                image = cv2.imread(image_path)
                detections = self.detector.predict_with_classes(
                    image=image,
                    classes=enhance_class_name(class_names=classes[:-1]),
                    box_threshold=box_threshold,
                    text_threshold=text_threshold
                )

                detections.mask = segment(
                    sam_predictor=self.segmentor,
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    xyxy=detections.xyxy
                )   

                boxes = torch.tensor(detections.xyxy)
                scores = torch.tensor(detections.confidence)

                keep_inds = nms(boxes, scores, 0.99).cpu().numpy()

                detections.xyxy = detections.xyxy[keep_inds]
                detections.mask = detections.mask[keep_inds]
                detections.confidence = detections.confidence[keep_inds]
                detections.class_id = detections.class_id[keep_inds]

                class_ids = detections.class_id
                class_ids[class_ids == None] = len(classes) - 1
                class_ids = class_ids.astype(int)
                
                classes = np.array(classes)
                num_objects = len(detections)

                if num_objects == 0:
                    continue

                self.masks += np.split(detections.mask, num_objects)

                self.bboxes_xyxy += np.split(detections.xyxy, num_objects)

                self.class_names += classes[class_ids].tolist()

                self.confidences += detections.confidence.tolist()

                result_str = "\n".join([f"{classes[class_id]} : {len(class_ids[class_ids == class_id])}" for class_id in np.unique(class_ids)])
                print(f"results : \n {result_str}")

                for i in range(num_objects):

                    self.source_paths.append(image_path)

                    masked_image = image * np.concatenate([detections.mask[i][..., None],
                                detections.mask[i][..., None],
                                detections.mask[i][..., None]], axis = 2)
                
                    ymin, xmin, ymax, xmax = detections.xyxy[i].astype(int)
                    masked_image = masked_image[xmin:xmax, ymin:ymax, :]

                    cv2.imwrite("temp.jpg", cv2.resize(masked_image, (300,300)))
                    dis(im("temp.jpg"))

                    embedding = self.embeddor.predict(masked_image, return_embedding = True).detach().cpu().numpy()
                    
                    if self.embedding_matrix is None:
                        self.embedding_matrix = embedding[None, ...]
                    else:
                        self.embedding_matrix = np.concatenate([self.embedding_matrix, embedding[None, ...]])

                if viz_outputs == True:
                    plot_detections(image,
                                    detections,
                                    classes,
                                    in_window = in_window
                                    )
        
        self.save_data()
        
    def save_data(self):
        
        dataset_hash_key = " ".join(self.source_paths)        
        dataset_hash_name = get_string_hash(dataset_hash_key)
        
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

        save_object_to_file(data_object, save_path_details)
        save_object_to_file(self.embedding_matrix, save_path_matrix)


    def plot_results(self,
                     in_window = False):
        
            plot_detections(self.input_image,
                            self.detections,
                            self.prompt_classes,
                            in_window = in_window
                            )
            


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

