import os
import pandas as pd
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import IMAGE_SIZE, CLASS_NAMES


class CheXpertDataset(Dataset):

    def __init__(
        self,
        csv_file,
        image_root,
        train=False
    ):

        self.data = pd.read_csv(csv_file)
        self.image_root = image_root

        # Keep the existing U-Zeros policy
        self.data[CLASS_NAMES] = (
            self.data[CLASS_NAMES]
            .fillna(0)
            .replace(-1, 0)
        )

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        if train:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),

                transforms.RandomRotation(
                    degrees=5
                ),

                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.08
                ),

                transforms.ToTensor(),

                normalize
            ])

        else:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),

                transforms.ToTensor(),

                normalize
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = str(
            row["Path"]
        ).replace("\\", "/")

        for prefix in [
            "CheXpert-v1.0-small/",
            "CheXpert-v1.0/"
        ]:

            if image_path.startswith(prefix):

                image_path = image_path[
                    len(prefix):
                ]

                break

        full_path = os.path.join(
            self.image_root,
            image_path
        )

        with Image.open(full_path) as img:
            image = img.convert("RGB")

        image = self.transform(image)

        labels = torch.tensor(
            row[CLASS_NAMES].to_numpy(
                dtype="float32"
            ),
            dtype=torch.float32
        )

        return image, labels