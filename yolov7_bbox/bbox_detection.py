import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import torch
import argparse
import time
from pathlib import Path
from pkg_resources import packaging
from PIL import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import os 
from models2.experimental import attempt_load
from utils2.datasets import LoadStreams, LoadImages
from utils2.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

@torch.no_grad()
def detect(model, source, device):
    stride, imgsz, kpt_label = 32, 640, 3
    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    data = {}
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        # Process detections
        xyxy_list = []
        conf_list = []
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                #pred_name_lst = [names[int(c)] for c in det[:, -1]]
                im0_with_box = im0.copy()
                i = 0
                for *xyxy, conf, cls in reversed(det):
                    xyxy_np = np.array([i.detach().cpu().numpy() for i in xyxy])
                    xyxy_list.append(xyxy_np)
                    conf_list.append(conf.detach().cpu().item())
    return path, xyxy_list, conf_list
