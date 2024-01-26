# @title import library
import sys

sys.path.append("..")

import argparse
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import math
import torch.nn as nn
import os
import cv2
import pandas as pd
import torch
import random
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from torch import nn
from models import iresnet
#from tqdm.notebook import tqdm
from tqdm import tqdm
from models import magface
from load_model import load_test_model
from torch.nn import Parameter
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ToTensor,
    Resize,
    RandomRotation,
    RandomVerticalFlip,
    ColorJitter,
)


# @title Set Random Seed
def set_seed(seed=42, loader=None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    try:
        loader.sampler.generator.manual_seed(seed)
    except AttributeError:
        pass


# @title inference function
def train_transforms(examples):
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            RandomRotation(degrees=(-10, 10)),
            ColorJitter(brightness=0.2, hue=0.2),
            ToTensor(),
            normalize,
        ]
    )
    examples["pixel_values"] = [
        transform(image.convert("RGB")) for image in examples["image"]
    ]
    return examples


@torch.no_grad()
def inference(net, img, model_name="resnet"):
    size = 112 if model_name == "resnet" else 224
    img = cv2.resize(img, (size, size))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img.copy()).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net.eval()
    with torch.no_grad():
        feat = net(img.to(device))
    return feat.detach().cpu().numpy()


def get_embedding_dict(dog_path, net, model_name="resnet"):
    d_ff = defaultdict(list)
    for folder in tqdm(os.listdir(f"{dog_path}")):
        for img_path in os.listdir(f"{dog_path}/{folder}"):
            img = cv2.imread(f"{dog_path}/{folder}/{img_path}")[..., ::-1]
            emb = inference(net, img, model_name)
            d_ff[folder].append(emb)
    return d_ff


def load_csv(net):
    df_lsf = pd.read_csv("../datasets/cat_dog_class_csv/lsf2.csv")  # lsf2.csv
    df_ll = pd.read_csv("../datasets/cat_dog_class_csv/ll2.csv")  # ll2.csv
    df_ff = pd.read_csv("../datasets/cat_dog_class_csv/ff2.csv")  # ff2.csv
    df_fsl = pd.read_csv("../datasets/cat_dog_class_csv/fsl2.csv")  # fsl2.csv
    d_ff = get_embedding_dict("../datasets/Test/found", net)
    d_fsl = get_embedding_dict("../datasets/Test/synthetic_lost", net)
    d_ll = get_embedding_dict("../datasets/Test/lost", net)
    d_lsf = get_embedding_dict("../datasets/Test/synthetic_found", net)
    return df_lsf, df_ll, df_ff, df_fsl, d_ff, d_fsl, d_ll, d_lsf


if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, help="input backbone name")
    parser.add_argument("--header", type=str, help="input header name")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "resnet"
    backbone_name = args.backbone

    # backbone_path = f"../source/Res_mag_del2img_ExtraIG_replaced_backbone.pth"
    net = load_test_model.prepare_backbone(
        model_path="../source/",
        backbone_name=backbone_name,  # "Res_mag_del2img_ExtraIG_replaced_backbone.pth",
        num_of_classes=13135,
        device="cpu",
    )
    net.to(device)
    print('device:', device)

    """
	For each type of ads (lost, synthetic found, found, synthetic lost), 
	we use the resnet backbone map each image into the embedding vector, 
	and store it into a dictionary, the key refers to the query/answer id, 
	and the values refer to the embedding vector of each image under this query id/answer id
	"""
    df_lsf, df_ll, df_ff, df_fsl, d_ff, d_fsl, d_ll, d_lsf = load_csv(net)

    """
	- Getting an the answers database embedding
	- In this part is synthetic found answers databaes, 
	- The shape of he database embedding: [number of images in the synthetic found database, 512]
	- And the length of the labels = number of images, and it stored the synthetic found answer key
	- Which corresponding to the embedding.
	- the cls_of_labels store the classes of the answer id, its length also = number of images in synthetic found
	"""
    for i, (k, v) in enumerate(tqdm(d_lsf.items())):
        temp = torch.from_numpy(np.array(v)).squeeze(1)
        embs = temp if i == 0 else torch.vstack((embs, temp))
        label = [k] * len(v)
        labels = label if i == 0 else labels + label
    labels = np.array(labels)
    cls_of_labels = np.array(
        [df_lsf[df_lsf["folder"] == l]["cls"].iloc[0] for l in labels]
    )

    embs.to(device)
    data_size = len(labels)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

    ans = []
    for q_name, q_embs in tqdm(d_ll.items()):
        q_emb = torch.from_numpy(np.array(q_embs).squeeze(1))
        cls_of_q = df_ll[df_ll["folder"] == q_name]["cls"].iloc[
            0
        ]  # founding the query class
        mask = (
            cls_of_labels == cls_of_q
        )  # create a mask for filtering the irrelevant answer
        # q_emb with size [number of images in the query, 512]
        for i, q in enumerate(q_emb):
            cos_sim = (
                cos(q, embs[mask])
                if i == 0
                else torch.hstack((cos_sim, cos(q, embs[mask])))
            )
            gts = labels[mask] if i == 0 else np.hstack((gts, labels[mask]))
        inds = torch.argsort(
            cos_sim, descending=True
        )  # argsort the cos_sim, it could be having some repeat answer.
        sorted_cos = []
        sorted_gts = []
        for c, g in zip(cos_sim[inds], gts[inds]):  # take top 100 answer without repeat
            if c not in sorted_cos:
                sorted_cos.append(c.item())
                sorted_gts.append(g)
            if len(sorted_cos) == 100:
                break

        ans.append(
            [q_name, sorted_cos[0], sorted_cos[2], sorted_cos[9], ",".join(sorted_gts)]
        )

    # same as the previous cell, but for found and synthetic lost.
    for i, (k, v) in enumerate(tqdm(d_fsl.items())):
        temp = torch.from_numpy(np.array(v)).squeeze(1)
        embs = temp if i == 0 else torch.vstack((embs, temp))
        label = [k] * len(v)
        labels = label if i == 0 else labels + label
    labels = np.array(labels)

    cls_of_labels = np.array(
        [df_fsl[df_fsl["folder"] == l]["cls"].iloc[0] for l in labels]
    )

    embs.to(device)
    data_size = len(labels)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
    for q_name, q_embs in tqdm(d_ff.items()):
        q_emb = torch.from_numpy(np.array(q_embs).squeeze(1))
        cls_of_q = df_ff[df_ff["folder"] == q_name]["cls"].iloc[0]
        mask = cls_of_labels == cls_of_q
        for i, q in enumerate(q_emb):
            cos_sim = (
                cos(q, embs[mask])
                if i == 0
                else torch.hstack((cos_sim, cos(q, embs[mask])))
            )
            gts = labels[mask] if i == 0 else np.hstack((gts, labels[mask]))
        inds = torch.argsort(cos_sim, descending=True)
        sorted_cos = []
        sorted_gts = []
        for c, g in zip(cos_sim[inds], gts[inds]):
            if c not in sorted_cos:
                sorted_cos.append(c.item())
                sorted_gts.append(g)
            if len(sorted_cos) == 100:
                break

        ans.append(
            [q_name, sorted_cos[0], sorted_cos[2], sorted_cos[9], ",".join(sorted_gts)]
        )

    ans_df = pd.DataFrame(
        data=ans, columns=["query", "matched_1", "matched_3", "matched_10", "answer"]
    )
    ans_df.to_csv("submit.tsv", sep="\t", index=False)
