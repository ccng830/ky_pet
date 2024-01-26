import sys

sys.path.append("..")

import os
import random
import numpy as np
import shutil
import json
import argparse
from collections import defaultdict
#from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from termcolor import cprint
import torch
import torch.backends.cudnn as cudnn
from models import magface
from torch import nn
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_

# our modules
from dataloader import dataloader
from load_model import load_model
from config import config as cfg


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


def save_log_file(save_path, train_log, val_log):
    with open(f"{save_path}/train_log.json", "w") as f:
        json.dump(train_log, f, indent=4)

    with open(f"{save_path}/val_log.json", "w") as f:
        json.dump(val_log, f, indent=4)


if __name__ == "__main__":
    set_seed(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_name", type=str, default="testing", help="input exp_name"
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="../datasets/All3img_ExtraIG_train",
        help="input train_data_path",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="../datasets/All3img_ExtraIG_val",
        help="input val_data_path",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="input batch_size")
    parser.add_argument("--num_epoch", type=int, default=30, help="input num_epoch")
    parser.add_argument(
        "--continue_train", action="store_true", help="input True or False"
    )
    parser.add_argument(
        "--backbone", type=str, help="input backbone name"
    )
    parser.add_argument(
        "--header", type=str, help="input header name"
    )

    args = parser.parse_args()
    exp_name = args.exp_name
    continue_train = args.continue_train
    train_data_path = args.train_data_path
    val_data_path = args.val_data_path
    batch_size = args.batch_size
    backbone_name = args.backbone
    header_name = args.header

    save_path = f"../myexp/{exp_name}"  #'testing'
    print(f"save path: {save_path}")
    print(f"batch size: {batch_size}")
    print(f"continue_train: {continue_train}")

    train_dataloader, val_dataloader, train_dataset, val_dataset = dataloader.prepare_dataloader(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        batch_size=batch_size,
    )
    num_of_classes = len(train_dataset.label_names)
    model = load_model.train(
        continue_train=continue_train,
        exp_name=exp_name,
        backbone_name=backbone_name,
        header_name=header_name,
        num_of_classes=num_of_classes
    )
    model.to(device)

    criterion = magface.MagLoss(
        cfg.config["la"],
        cfg.config["ua"],
        cfg.config["l_margin"],
        cfg.config["u_margin"],
    )

    opt_backbone = torch.optim.SGD(
        params=[{"params": model.features.parameters()}],
        lr=cfg.config["lr"] / 512 * batch_size,
        momentum=0.9,
        weight_decay=cfg.config["weight_decay"],
    )
    opt_header = torch.optim.SGD(
        params=[{"params": model.fc.parameters()}],
        lr=cfg.config["lr"] / 512 * batch_size,
        momentum=0.9,
        weight_decay=cfg.config["weight_decay"],
    )

    if not os.path.exists(f"{save_path}"):
        os.makedirs(f"{save_path}")

    if not continue_train:
        print()
        print("Training start...!!:")
        train_log = defaultdict(list)
        val_log = defaultdict(list)
    else:
        print()
        print("Training continue...!!:")
        with open(f"{save_path}/train_log.json") as f:
            train_log = json.load(f)
        with open(f"{save_path}/val_log.json") as f:
            val_log = json.load(f)

    num_epoch = args.num_epoch
    start_epoch = 0 if not continue_train else len(train_log["acc"])
    min_loss = float("inf") if not continue_train else min(train_log["loss"])
    min_train_loss, min_val_loss = float("inf"), float("inf")
    max_val_acc = 0

    print("start_epoch:", start_epoch)
    print("num_epoch:", num_epoch)

    la, ua, l_margin, u_margin = 10, 110, 0.45, 0.8
    criterion = magface.MagLoss(la, ua, l_margin, u_margin)
    opt_backbone = torch.optim.SGD(
        params=[{"params": model.features.parameters()}],
        lr=cfg.config["lr"] / 512 * batch_size,
        momentum=0.9,
        weight_decay=cfg.config["weight_decay"],
    )
    opt_header = torch.optim.SGD(
        params=[{"params": model.fc.parameters()}],
        lr=cfg.config["lr"] / 512 * batch_size,
        momentum=0.9,
        weight_decay=cfg.config["weight_decay"],
    )

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.config["lr_func"]
    )
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=cfg.config["lr_func"]
    )

    for epoch in range(start_epoch, num_epoch):
        print(f"Runing exp: {exp_name}")
        print(f"Epoch: {epoch}")
        train_loss, train_correct, train_count = 0, 0, 0
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            label = batch["labels"].to(device)
            img = batch["pixel_values"].to(device)
            if len(img) == 1:
                print("batch size only have 1!")
                continue
            output, x_norm = model(img, label)
            loss_id, loss_g, one_hot = criterion(output, label, x_norm)
            loss = loss_id + 35 * loss_g  # lambda_g 35
            loss.backward()

            train_loss += loss.item()

            clip_grad_norm_(model.features.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()
            opt_backbone.zero_grad()
            opt_header.zero_grad()

            ans = torch.argmax(output[0], dim=1)
            train_correct += sum(ans == label)
            train_count += batch_size

            if idx % 100 == 0:
                print(f"after {idx} step loss: {train_loss/train_count}")
                print(f"acc: {train_correct/train_count}")

        train_log["loss"].append(train_loss / train_count)
        train_log["acc"].append((train_correct / train_count).item())
        print(f"Train_loss: {train_loss/train_count}")
        print("Train_acc:", train_correct / train_count)
        scheduler_backbone.step()
        scheduler_header.step()

        last_backbone_path = f"{save_path}/last_backbone.pth"
        torch.save(model.features.state_dict(), last_backbone_path)
        print(f"last_backbone save to: {last_backbone_path}")

        last_header_path = f"{save_path}/last_header.pth"
        torch.save(model.fc.state_dict(), last_header_path)
        print(f"last_header save to: {last_header_path}")

        val_loss, val_correct, val_count = 0, 0, 0
        model.eval()
        for idx, batch in enumerate(tqdm(val_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            label = batch["labels"].to(device)
            img = batch["pixel_values"].to(device)
            if len(img) == 1:
                print("batch size only have 1!")
                continue
            with torch.no_grad():
                output, x_norm = model(img, label)
                loss_id, loss_g, one_hot = criterion(output, label, x_norm)
                loss = loss_id + 35 * loss_g  # lambda_g 35

            val_loss += loss.item()

            clip_grad_norm_(model.features.parameters(), max_norm=5, norm_type=2)

            ans = torch.argmax(output[0], dim=1)
            val_correct += sum(ans == label)
            val_count += batch_size

            if idx % 100 == 0:
                print(f"after {idx} step loss: {val_loss/val_count}")
                print(f"acc: {val_correct/val_count}")

        val_log["loss"].append(val_loss / val_count)
        val_log["acc"].append((val_correct / val_count).item())
        print(f"val_loss: {val_loss/val_count}")
        print("val_acc:", val_correct / val_count)

        clear_output()

        val_acc = (val_correct / val_count).item()

        save_log_file(save_path, train_log=train_log, val_log=val_log)

        if val_acc > max_val_acc:
            max_val_acc = val_acc
            best_backbone_path = f"{save_path}/best_backbone.pth"
            torch.save(model.features.state_dict(), best_backbone_path)
            print(f"best_backbone save to: {best_backbone_path}")

            best_header_path = f"{save_path}/best_header.pth"
            torch.save(model.fc.state_dict(), best_header_path)
            print(f"best_header save to: {best_header_path}")

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_log["loss"], "x-", label="train loss")
        plt.plot(val_log["loss"], "x-", label="val loss")
        plt.grid()
        plt.legend()
        plt.xlabel("epoch")

        plt.subplot(1, 2, 2)
        plt.plot(train_log["acc"], "rx-", label="train acc")
        plt.plot(val_log["acc"], "x-", label="val acc")
        plt.grid()
        plt.legend()
        plt.xlabel("epoch")

        plt.savefig(f"{save_path}/log.png")
