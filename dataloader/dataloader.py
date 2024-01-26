import sys
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip
from torchvision.transforms import ToTensor, RandomRotation
from torchvision.transforms import Resize, RandomVerticalFlip, ColorJitter


# @title Prepare Dataset
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform):
        print("data_path:", data_path)
        self.dataset_path = data_path
        self.images_path = []
        self.transform = transform
        self.label_names = sorted(list(set(os.listdir(self.dataset_path))))
        self.label2id = {name: i for i, name in enumerate(self.label_names)}
        self.id2label = {i: name for i, name in enumerate(self.label_names)}

        self.labels = []
        for label in os.listdir(self.dataset_path):
            for img_path in os.listdir(f"{self.dataset_path}/{label}"):
                self.images_path.append(f"{self.dataset_path}/{label}/{img_path}")
                self.labels.append(self.label2id[label])

    def _get_train_item(self, idx):
        image = self.transform(Image.open(self.images_path[idx]).convert("RGB"))
        label = self.labels[idx]
        return {
            "pixel_values": image,
            "label": label,
        }

    def __getitem__(self, idx):
        return self._get_train_item(idx)

    def __len__(self):
        return len(self.images_path)


# @title Prepare dataloader
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def prepare_dataloader(
    train_data_path="../datasets/All3img_ExtraIG_train",
    val_data_path="../datasets/All3img_ExtraIG_val",
    batch_size=16,
):
    normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    train_transform = Compose(
        [
            Resize((112, 112)),
            RandomHorizontalFlip(0.1),
            RandomVerticalFlip(0.05),
            RandomRotation(degrees=(-10, 10)),
            ColorJitter(brightness=0.2, hue=0.2),
            ToTensor(),
            normalize,
        ]
    )
    val_transform = Compose([Resize((112, 112)), ToTensor(), normalize])
    train_dataset = TrainDataset(train_data_path, train_transform)
    val_dataset = TrainDataset(val_data_path, val_transform)
    train_dataloader = DataLoader(
        train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True
    )

    print(
        "number_of_classes:", len(train_dataset.label_names)
    )  # number_of_classes = 13135
    print("load dataset success!")
    return train_dataloader, val_dataloader, train_dataset, val_dataset
