import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import torchvision.transforms as transforms

from pytorchyolo import detect, models

from pytorchyolo.models import load_model_own
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from pytorchyolo.utils.datasets import ImageFolder
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS


from utils.viz_utils import create_frame
from utils.viz_utils import draw_bounding_boxes
from utils.model_utils import custom_detections
"""
let yolo make detections, and make embeddings using the detections class, image, and the embedding matrix like resnet_embedder
- test embedding augmenter a bit more,
- make the yolo_embedder_class, such that it takes the same inputs as resnet_embedder, and outputs the same as well.
"""

class yolo_model:
    def __init__(self):
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
                    conf_thres=0.5,
                    nms_thres=0.5,
                    ):
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
    
    def predict_with_classes(self,
                    img,
                    box_threshold=0.5,
                    nms_thres=0.5,
                    text_threshold = None ): # text_threshold is a dummy attribute, to maintain a standard predict function
        
        conf_thres = box_threshold
        img_size=416
    
        detections = self.predict_raw(img,
                                      img_size=img_size,
                                      conf_thres=conf_thres,
                                      nms_thres=nms_thres
                                      )

        boxes = non_max_suppression(boxes, conf_thres, nms_thres)
        boxes = rescale_boxes(boxes[0], img_size, img.shape[:2])
        boxes = boxes.numpy()

        detections = custom_detections(xyxy = boxes[:, :4],
                                       masks = None,
                                       confidence = boxes[:, 4],
                                       class_ids= boxes[:, 5]
                                       )

        return detections    
    
    def predict_boxes(self,
                    img,
                    img_size=416,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    head= -1,
                    viz = False,
                    reverse_img_for_viz = True):
        
        detections = self.predict_raw(img,
                                      img_size=img_size,
                                      conf_thres=conf_thres,
                                      nms_thres=nms_thres
                                      )
    
        if isinstance(head, int):
            if head == -1:
                boxes = torch.cat(detections[1], 1).clone()
            elif head < 3:
                boxes = detections[1][head]
            else:
                print(f"\n YOLOV3 HAS ONLY 3 HEADS \n")
                return

        else:
            print(f"\n passed : ' {head} ': THE 'head' ARGUEMENT CORRESPONDS TO HEAD INDEX FROM WHICH YOU WANT THE PREDICTIONS,")
            print("\n\n-1 CORRESPONDS TO ALL HEADS\n")
            return    

        boxes = non_max_suppression(boxes, conf_thres, nms_thres)
        # print(f"num boxes post supression : {head_detections[0].shape}")
        boxes = rescale_boxes(boxes[0], img_size, img.shape[:2])
        boxes = boxes.numpy()

        if viz:
            image = draw_bounding_boxes(img, boxes, self.classes)
            if reverse_img_for_viz:
                image = image[:, :, ::-1]
            plt.imshow()
            plt.show()
        
        return image, boxes    


    def get_embedding(self,
                      img,
                      img_size=416,
                      conf_thres=0.5,
                      nms_thres=0.5,
                      head = 0,
                      embedding_window_size = 5,
                      viz = True):
        
        
        detections = self.predict_raw(img,
                                      img_size=img_size,
                                      conf_thres=conf_thres,
                                      nms_thres=nms_thres
                                      )

        if head > 2:
            print("VALUE ERROR you passed ' {head} ' YOLOV3 HAS ONLY 3 HEADS\n")
            return -1
        elif head > 1:
            print(f"NOT IMPLEMENTED ERROR you passed ' {head} ', only head = 0 is implemented")
            return -1
        head_inds = np.nonzero(np.array([module_def["type"] == "yolo" for module_def in self.model.module_defs]))

        embeddding_layer_ind = head_inds[head] - 2

        raw_embeddings = detections[0][embeddding_layer_ind]

        image, boxes = self.predict_boxes(img,
                    img_size=416,
                    conf_thres=0.5,
                    nms_thres=0.5,
                    head= head,
                    viz = viz,
                    reverse_img_for_viz = True)

        
        batch_size, num_channels , grid_num_rows, grid_num_cols = detections[0][embeddding_layer_ind].shape 

        grid_dim_row = int(round(img.shape[0] / grid_num_rows))
        grid_dim_col = int(round(img.shape[1] / grid_num_cols))
        grid_dims = np.array([grid_dim_col, grid_dim_row])

        centers = (boxes[:, 2:4] - boxes[: , :2])/2 + boxes[: , :2]

        mem_starts_rowcol = centers[:, ::-1] // grid_dims[::-1]

        embedding_slice_min = np.maximum(mem_starts_rowcol - 2, np.zeros_like(mem_starts_rowcol)).astype(int)
        embedding_slice_max = np.minimum(mem_starts_rowcol + 3, np.ones_like(mem_starts_rowcol) * (grid_dims[::-1] )).astype(int)

        viz_embedding = np.concatenate([embedding_slice_min, embedding_slice_max], axis = 1)

        print(viz_embedding)
        print(mem_starts_rowcol)

        raw_embedding = detections[0][80].detach().cpu().numpy()

                
        # #XXXXXXXXXXXXXXXXXXXXXXXXXX     video details
        # frames,
        # output_path = '/app/bin/outputs/embedding_occulsion_analysis_head_1.mp4'
        output_path = '/app/bin/outputs/embedding_assignment_method.mp4'
        fps=0.4
        frame_size=(800, 400)

        frame_counter = 0

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Using 'mp4v' for compatibility
        video = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        # #XXXXXXXXXXXXXXXXXXXXXXXXXXX

        embeddings = []
        for ind, (rowmin, colmin, rowmax, colmax) in enumerate(viz_embedding.astype(int)):

            if viz == True:
                
                vizimg = img.copy()
                vizimg = cv2.rectangle(vizimg, (colmin * grid_dim_row, rowmin * grid_dim_col) , (colmax * grid_dim_row , rowmax * grid_dim_col), (0,0,0) , 15)
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
                video.write(resized_frame)

                # plt.imshow(frame[:, :, ::-1])
                # plt.show()

        video.release()

        return embeddings
    

# class yolo_embedder:
#     def



        