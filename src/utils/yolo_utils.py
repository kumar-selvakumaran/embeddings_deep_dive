
"""
author = "Kumar Selvakumaran", "Mrudula Acharya", "Neel Adke"
date = "04/24/2024"

Module Docstring
This file contains the  required functions and classes to load, do inference ,and extract embeddings using the YOLOv3 model 
"""
import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import torchvision.transforms as transforms
from torchvision.ops import nms,  clip_boxes_to_image

from pytorchyolo import detect, models

from pytorchyolo.models import load_model_own
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from pytorchyolo.utils.datasets import ImageFolder
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

from .data_utils import get_string_hash, save_object_to_file

from utils.viz_utils import create_frame
from utils.viz_utils import draw_bounding_boxes

"""mirror the object embeding to satisfy window size (mirroring from edge, not center)"""
def augment_embedddings(embedding,
                        embedding_border, #tuple([rowmin, colmin, rowmax, colmax])
                        total_embedding_dims,
                        embedding_window_size = 5): #tuple([num rows , num cols]) in image embedding 
    """
    Augment the embeddings by mirroring the edges if the designated window is smaller than the required embedding window size.

    Parameters:
    embedding (np.array): The original embedding matrix.
    embedding_border (tuple): The bounds of the embedding in the format (rowmin, colmin, rowmax, colmax).
    total_embedding_dims (tuple): The total dimensions of the embedding matrix as (rows, cols).
    embedding_window_size (int, optional): The size of the window for the embedding in terms of number of rows and columns. Default is 5.

    Returns:
    np.array: The augmented embedding matrix.
    """
    rowmin, colmin, rowmax, colmax = embedding_border
    num_missing_rows = embedding_window_size - (rowmax - rowmin)
    num_missing_cols = embedding_window_size - (colmax - colmin)
    max_embedding_row, max_embedding_col = total_embedding_dims
    
    changed  = False

    if num_missing_rows>0:
        print(f"AUGMENTING EMBEDDINGS\n\noriginal_embedding : {(embedding[0, 0, :, :]*1000).astype(int)}, rowmin : {rowmin} ,  colmin : {colmin}, rowmax : {rowmax}, colmax : {colmax}\n")
        if rowmin == 0:
            print("augmenting top border")
            # aug_emb = embedding[rowmax-num_missing_rows:][::-1, :, :]
            aug_emb = embedding[:, :, 1:num_missing_rows+1, :][:, :, ::-1, :]
            print(embedding.shape, aug_emb.shape)
            embedding = np.concatenate([aug_emb, embedding], axis = 2)
        elif rowmax == max_embedding_row:
            print("augmenting bottom border")
            # aug_emb = embedding[:num_missing_rows][::-1, :, :]
            aug_emb = embedding[:, :, ::-1, :][:, :, 1:num_missing_rows+1, :]
            print(embedding.shape, aug_emb.shape)
            embedding = np.concatenate([embedding, aug_emb], axis = 2)

        changed = True
    
    if num_missing_cols > 0:
        print(f"AUGMENTING EMBEDDINGS\n\noriginal_embedding : {(embedding[0, 0, :, :]*1000).astype(int)}, rowmin : {rowmin} ,  colmin : {colmin}, rowmax : {rowmax}, colmax : {colmax}\n")
        if colmin == 0:
            print("augmenting left border")
            aug_emb = embedding[:, :, :, 1:num_missing_cols+1][:, :, :, ::-1]
            print(embedding.shape, aug_emb.shape)
            embedding = np.concatenate([aug_emb, embedding], axis = 3)
        if colmax == max_embedding_col:
            print("augmenting right border")
            aug_emb = embedding[:, :, :, ::-1][:, :, :, 1:num_missing_cols+1]
            print(embedding.shape, aug_emb.shape)
            embedding = np.concatenate([embedding, aug_emb], axis = 3)
        
        changed = True

    if changed == True:
        print(f"augmented embedding : {(embedding[0, 0, :, :]*1000).astype(int)}")
        
    return embedding

class yolo_model:
    """
    A class for handling object detection tasks using a YOLOv3 model pre-trained on the COCO dataset.

    Attributes:
    classes (list): List of class names from the COCO dataset.
    model (torch model): The loaded YOLOv3 model.
    device (torch.device): The computation device (CUDA if available).
    """
    def __init__(self):
        """
        Initializes the yolo_model class by loading the class names, model configuration, and weights, and setting the device.
        """
        with open("/app/PyTorch-YOLOv3/data/coco.names", 'r') as rf:
            self.classes = rf.read().split('\n')

        self.model = models.load_model_own(
            "/app/PyTorch-YOLOv3/config/yolov3.cfg",
            "/app/bin/model_files/yolov3.weights")
        
        self.device = torch.device('cuda')

        self.model.to(self.device)

    def predict_raw(self,
                    img,
                    img_size=416,
                    conf_thres=0.5):
        """
        Runs a forward pass to predict objects in an image without applying non-maximum suppression.

        Parameters:
        img (numpy array): The input image in BGR format.
        img_size (int): The size to which the images are resized.
        conf_thres (float): Confidence threshold to filter detections.

        Returns:
        torch.Tensor: Raw model detections.
        """
        self.model.eval()
                
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        input_img = transforms.Compose([
            DEFAULT_TRANSFORMS,
            Resize(img_size)])(
                (img, np.zeros((1, 5))))[0].unsqueeze(0)

        input_img = input_img.to(self.device)

        with torch.no_grad():
            detections = self.model(input_img)

        return detections
    

    def predict_boxes(self,
                    img,
                    img_size=416,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    head= -1,
                    viz = False,
                    reverse_img_for_viz = True):
        """
        Predicts bounding boxes for objects detected in the image and optionally visualizes the results.

        Parameters:
        img (numpy array): The input image in BGR format.
        img_size (int): The size to which the images are resized.
        conf_thres (float): Confidence threshold to filter detections.
        nms_thres (float): Non-maximum suppression threshold.
        head (int): The specific head of the YOLO model to use.
        viz (bool): Whether to visualize the results.
        reverse_img_for_viz (bool): Whether to convert images back to BGR for visualization.

        Returns:
        tuple: A tuple containing the image with drawn bounding boxes and the list of bounding boxes.
        """
        
        detections = self.predict_raw(img,
                                      img_size=img_size,
                                      conf_thres=conf_thres)
    
        if isinstance(head, int):
            if head == -1:
                boxes = torch.cat(detections[1], 1).clone()
            elif head < 3:
                boxes = detections[1][head]
            else:
                print(f"\n YOLOV3 HAS ONLY 3 HEADS \n")
                return

        else:
            print(f"\n passed : ' {head} ': THE 'head' ARGUEMENT CORRESPONDS TO HEAD INDEX FROM WHICH YOU WANT THE PREDICTIONS,\n\n-1 CORRESPONDS TO ALL HEADS\n")
            return    

        #nms intra class
        boxes = non_max_suppression(boxes, conf_thres, nms_thres)[0]

        #nms inter class
        keep_inds = nms(boxes[:, :4], boxes[:, 4], nms_thres).cpu().numpy()
        boxes = boxes[keep_inds]


        # print(f"num boxes post supression : {head_detections[0].shape}")
        boxes = rescale_boxes(boxes, img_size, img.shape[:2])
        
        boxes = clip_boxes_to_image(boxes, tuple(img.shape[:2]))
        
        boxes = boxes.numpy()
        image = draw_bounding_boxes(img, boxes, self.classes)
        if viz:
            if reverse_img_for_viz:
                image = image[:, :, ::-1]
            plt.imshow(image)
            plt.show()
        return image, boxes    


    def get_embeddings(self,
                      image_paths,
                      img_size=416,
                      conf_thres=0.5,
                      nms_thres=0.5,
                      head = 0,
                      embedding_window_size = 5,
                      viz = True,
                      save = True,
                      save_str = 'default'):
        """
        Retrieves embeddings for detected objects in a series of images and optionally saves and visualizes them.

        Parameters:
        image_paths (list): Paths to the images.
        img_size (int): The size to which the images are resized.
        conf_thres (float): Confidence threshold to filter detections.
        nms_thres (float): Non-maximum suppression threshold.
        head (int): The specific head of the YOLO model to use for embeddings.
        embedding_window_size (int): The window size around the detected object for extracting embeddings.
        viz (bool): Whether to visualize the results.
        save (bool): Whether to save the embeddings.
        save_str (str): A string to append to the save filename for uniqueness.

        Returns:
        tuple: A tuple containing the embedding details and the embedding matrix.
        """
        
        
        # #XXXXXXXXXXXXXXXXXXXXXXXXXX     video details
        # frames,
        # output_path = '/app/bin/outputs/embedding_occulsion_analysis_head_1.mp4'
        output_path = '/app/bin/outputs/getting_embeddings.mp4'
        fps=0.6
        frame_size=(800, 400)

        frame_counter = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using 'mp4v' for compatibility
        video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        # #XXXXXXXXXXXXXXXXXXXXXXXXXXX

        if head > 2:
            print("VALUE ERROR you passed ' {head} ' YOLOV3 HAS ONLY 3 HEADS\n")
            return -1
        elif head > 1:
            print(f"NOT IMPLEMENTED ERROR you passed ' {head} ', only head = 0 is implemented")
            return -1
        head_inds = np.nonzero(np.array([module_def["type"] == "yolo" for module_def in self.model.module_defs]))[0]

        embedding_layer_ind = head_inds[head] - 2
                    
        embeddings = []

        all_boxes = []
        source_paths = []
        class_names = []
        confidences = []
        # embeddding matrix is made in the end

        for image_path in image_paths:
            img = cv2.imread(image_path)
            detections = self.predict_raw(img,
                                        img_size=img_size,
                                        conf_thres=conf_thres)

            raw_embeddings = detections[0][embedding_layer_ind]

            image, boxes = self.predict_boxes(img,
                        img_size=img_size,
                        conf_thres=conf_thres,
                        nms_thres=nms_thres,
                        head= head,
                        viz = viz,
                        reverse_img_for_viz = True)

            if len(boxes) == 0:
                print("\n NO DETECTIONS \n")
                print(f"\b MOVING TO THE NEXT IMAGES \n")
                continue

            # appending embedding details 
            all_boxes += np.split(boxes[:, :4], len(boxes))
            source_paths += [image_path for i in range(len(boxes))]
            class_names += np.array(self.classes)[boxes[:, 5].astype(int)].tolist()
            confidences += boxes[:, 4].tolist()


            batch_size, num_channels , grid_num_rows, grid_num_cols = detections[0][embedding_layer_ind].shape 

            grid_dim_row = int(round(img.shape[0] / grid_num_rows))
            grid_dim_col = int(round(img.shape[1] / grid_num_cols))
            grid_dims = np.array([grid_dim_col, grid_dim_row])

            centers = (boxes[:, 2:4] - boxes[: , :2])/2 + boxes[: , :2]

            mem_starts_rowcol = centers[:, ::-1] // grid_dims[::-1]

            # embedding_slice_min = np.maximum(mem_starts_rowcol - 2, np.zeros_like(mem_starts_rowcol)).astype(int)
            # embedding_slice_max = np.minimum(mem_starts_rowcol + 3, np.ones_like(mem_starts_rowcol) * (grid_dims[::-1] )).astype(int)

            slice_ind_min, slice_ind_max = math.floor(embedding_window_size / 2) , math.ceil(embedding_window_size / 2) 
            embedding_slice_min = np.maximum(mem_starts_rowcol - slice_ind_min, np.zeros_like(mem_starts_rowcol)).astype(int)
            embedding_slice_max = np.minimum(mem_starts_rowcol + slice_ind_max, np.ones_like(mem_starts_rowcol) * (grid_dims[::-1] )).astype(int)
            

            object_embedding_locations = np.concatenate([embedding_slice_min, embedding_slice_max], axis = 1)

            print(f" object embedding_locations : {object_embedding_locations}")
            # print(mem_starts_rowcol)

            raw_embedding = raw_embeddings.detach().cpu().numpy()

            for ind, (rowmin, colmin, rowmax, colmax) in enumerate(object_embedding_locations.astype(int)):
                object_embedding = raw_embedding[:, :, rowmin:rowmax, colmin:colmax].copy()
                object_embedding = augment_embedddings(object_embedding, (rowmin, colmin, rowmax, colmax), (grid_num_rows, grid_num_cols), embedding_window_size)
                embeddings.append(object_embedding.flatten())
                

                if viz == True:
                    
                    vizimg = img.copy()
                    vizimg = cv2.rectangle(vizimg, (colmin * grid_dim_col, rowmin * grid_dim_row) , (colmax * grid_dim_col , rowmax * grid_dim_row), (0,0,0) , 15)
                    # vizimg = cv2.rectangle(vizimg, (colmin * grid_dim_row, rowmin * grid_dim_col) , (colmax * grid_dim_row , rowmax * grid_dim_col), (0,0,0) , 15)
                    vizimg = draw_bounding_boxes(vizimg, boxes[ind][None, ...], self.classes)
                    center_x , center_y = centers[ind].astype(int)
                    vizimg = cv2.circle(vizimg, (center_x, center_y), 20, (0,0,0), 22)
                    vizimg = cv2.circle(vizimg, (center_x, center_y), 20, (255,255,255), 5)

                    viz_emb = raw_embedding[0, 0, :, :].copy()
                    r, c = mem_starts_rowcol[ind].astype(int)
                    viz_emb[rowmin:rowmax, colmin:colmax] = 0.0
                    viz_emb[r,c] = 0.75
                        
                    frame = create_frame([viz_emb, vizimg], ["embedding", "results"], figsize=(10, 5))
                    
                    resized_frame = cv2.resize(frame, frame_size)  
                    video.write(resized_frame[:, :, ::-1])
                    plt.imshow(frame)
                    plt.show()
        
        print(f"releasing video")
        video.release()

        embedding_details = {
            'bounding_boxes' : all_boxes,
            'confidences' : confidences,
            'class_names' : class_names,
            'source_paths' : source_paths,
        } 

        save_str = f"numims_{len(image_paths)}_embwin_{embedding_window_size}"
        
        self.save_embedding_details(embedding_details, np.array(embeddings), save_str)

        return embedding_details , np.array(embeddings) 
    
    def save_embedding_details(self,
                               embedding_details,
                               embedding_matrix,
                               save_str = 'default'):
        """
        Saves the embedding details and matrix to file.

        Parameters:
        embedding_details (dict): Details of the embeddings including bounding boxes and class names.
        embedding_matrix (numpy array): The embedding matrix.
        save_str (str): A string to append to the save filename for uniqueness.
        """
        dataset_hash_key = " ".join(embedding_details['source_paths']) 
        dataset_hash_name = get_string_hash(dataset_hash_key)  + f"_yolo_{save_str}"   

        save_path_details = os.path.join("/app/bin/results/" , f'embedding_details_{dataset_hash_name}.pkl')
        save_path_matrix = os.path.join("/app/bin/results/" , f'embedding_matrix_{dataset_hash_name}.pkl')
        
        embedding_details["embedding_matrix_path"] = save_path_matrix

        
        print(f"\nsaved file name : {dataset_hash_name}\n")
        save_object_to_file(embedding_details, save_path_details)
        save_object_to_file(embedding_matrix, save_path_matrix)

        
    
    
    