import sys
sys.path.append('yolov7_bbox')

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
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

@torch.no_grad()
def detect(opt, source):
    weights, imgsz, kpt_label = opt.weights, opt.img_size, opt.kpt_label

    dataset = LoadImages(source, img_size=imgsz, stride=stride)
    data = {}
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=None, agnostic=False)

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
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                pred_name_lst = [names[int(c)] for c in det[:, -1]]
                im0_with_box = im0.copy()
                i = 0
                for *xyxy, conf, cls in reversed(det):
                    xyxy_np = np.array([i.detach().cpu().numpy() for i in xyxy])
                    xyxy_list.append(xyxy_np)
                    conf_list.append(conf.detach().cpu().item())
    return path, xyxy_list, conf_list
        

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
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())
    imgsz = check_img_size(640, s=stride)
    model = TracedModel(model, device, 640)
    names = model.module.names if hasattr(model, 'module') else model.names[::-1]

    data = {}
    # folder of images
    if any([path.lower().endswith(('.jpg', '.png', '.jpeg', '.webp', '.gif', '.bmp')) for path in os.listdir(source)]):
        for path in os.listdir(source):
            path_name, bbox, conf = detect(opt=opt, source=f"{source}/{path}")
            print("Done", path_name)
            data[path_name] = {
                'bbox': [i.tolist() for i in bbox],
                'folder': '',
                'conf': conf
                }
    else: # folder of folder of images
        for folder in os.listdir(source):
            for img_path in os.listdir(f"{source}/{folder}"):
                try:
                    path_name, bbox, conf = detect(opt=opt, source=f"{source}/{folder}/{img_path}")
                    print("Done", path_name)
                    data[path_name] = {
                        'bbox': [i.tolist() for i in bbox],
                        'folder': folder,
                        'conf': conf
                        }
                except:
                    print(f"fail on {img_path}")
        
    save_path = f"{opt.name}_bbox.json"
    if os.path.exists(save_path):
        num = len([i for i in os.listdir('.') if f"{opt.name}_bbox" in i])
        save_path = f"{opt.name}_bbox{num}.json"

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
    print("bbox data save to:", save_path)
