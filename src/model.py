import torch.nn as nn
from torchvision import models

from src.config import NUM_CLASSES


def create_model(model_name="resnet18"):

    if model_name == "resnet18":

        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        model.fc = nn.Linear(
            model.fc.in_features,
            NUM_CLASSES
        )

    elif model_name == "densenet121":

        model = models.densenet121(
            weights=models.DenseNet121_Weights.DEFAULT
        )

        model.classifier = nn.Linear(
            model.classifier.in_features,
            NUM_CLASSES
        )

    else:
        raise ValueError(
            f"Unsupported model: {model_name}"
        )

    return model