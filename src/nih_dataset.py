import os
from pathlib import Path

import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import IMAGE_SIZE


NIH_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Effusion"
]


class NIHDataset(Dataset):

    def __init__(
        self,
        csv_file="datasets/nih/Data_Entry_2017_v2020.csv",
        image_root="datasets/nih/images",
        image_names=None
    ):

        self.data = pd.read_csv(csv_file)

        if image_names is not None:
            self.data = self.data[
                self.data["Image Index"].isin(image_names)
            ].copy()

        self.data = self.data.reset_index(drop=True)

        # Build labels using exact matching.
        # NIH uses pipe-separated finding names.
        findings = (
            self.data["Finding Labels"]
            .fillna("")
            .str.split("|")
        )

        self.labels = torch.tensor(
            [
                [
                    float(disease in labels)
                    for disease in NIH_CLASSES
                ]
                for labels in findings
            ],
            dtype=torch.float32
        )

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Find images recursively.
        self.image_paths = {}

        for root, _, files in os.walk(image_root):
            for filename in files:
                if filename.lower().endswith(".png"):
                    self.image_paths[filename] = os.path.join(
                        root, filename
                    )

        missing = [
            name
            for name in self.data["Image Index"]
            if name not in self.image_paths
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} NIH images are missing. "
                f"Example: {missing[:5]}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_name = row["Image Index"]

        image_path = self.image_paths[image_name]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        labels = self.labels[index]

        return image, labels