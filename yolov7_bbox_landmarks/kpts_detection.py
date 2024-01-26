import argparse
import time
from pathlib import Path
import matplotlib.pyplot as plt
import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import json

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh, check_requirements
from utils.torch_utils import select_device
import numpy as np
import pandas as pd

def load_model(weights):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = attempt_load(weights, map_location=device)
	return model

@torch.no_grad()
def detect(model, source, device):
    stride, imgsz, kpt_label = 32, 640, 3
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]
        print(pred[...,4].max())
        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, kpt_label=kpt_label)

        # Process detections
        xyxy_list = []
        src_pts_list = []
        conf_list = []
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    kpts = det[det_index, 6:]
                    src_pts = kpts.detach().cpu().numpy()[[6, 7, 3, 4, 0, 1]]
                    xyxy_np = np.array([i.detach().cpu().numpy() for i in xyxy])
                    
                    xyxy_list.append(xyxy_np)
                    src_pts_list.append(src_pts)
                    conf_list.append(conf.detach().cpu().item())

            print(f"Done: {path}")
    return path, xyxy_list, src_pts_list, conf_list