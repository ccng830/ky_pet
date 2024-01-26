import sys
sys.path.append('yolov7_bbox_landmarks')

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

@torch.no_grad()
def detect(opt, source):
    weights, imgsz, kpt_label = opt.weights, opt.img_size, opt.kpt_label

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
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=False, kpt_label=kpt_label)

        # Process detections
        xyxy_list = []
        src_pts_list = []
        conf_list = []
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

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
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-pet-face.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source, pass images or folder') 
    parser.add_argument('--img-size', nargs= '+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--kpt-label', type=int, default=3, help='number of keypoints')
    opt = parser.parse_args()
    print(opt)

    source, weights, imgsz, kpt_label = opt.source, opt.weights, opt.img_size, opt.kpt_label
    device = select_device(opt.device)
    model = attempt_load(weights, map_location=device) 
    stride = int(model.stride.max()) 
    imgsz = check_img_size(640, s=stride)
    names = model.module.names if hasattr(model, 'module') else model.names

    data = {}
    # folder of images
    if any([path.lower().endswith(('.jpg', '.png', '.jpeg', '.webp', '.gif', '.bmp')) for path in os.listdir(source)]):
        for path in os.listdir(source):
            path_name, bbox, src_pts, conf = detect(opt=opt, source=f"{source}/{path}")
            data[path_name] = {
                'bbox': [i.tolist() for i in bbox],
                'kpts': [i.tolist() for i in src_pts],
                'folder': '',
                'conf': conf
                }

    else: # folder of folder of images
        for folder in os.listdir(source):
            for img_path in os.listdir(f"{source}/{folder}"):
                try:
                    path_name, bbox, src_pts, conf = detect(opt=opt, source=f"{source}/{folder}/{img_path}")
                    data[path_name] = {
                        'bbox': [i.tolist() for i in bbox],
                        'kpts': [i.tolist() for i in src_pts],
                        'folder': folder,
                        'conf': conf
                        }
                except:
                    print(f"fail on: {img_path}")
        
    save_path = f"{opt.name}_kpt.json"
    if os.path.exists(save_path):
        num = len([i for i in os.listdir('.') if f"{opt.name}_kpt" in i])
        save_path = f"{opt.name}_kpt{num}.json"

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
    print("kpts data save to:", save_path)
