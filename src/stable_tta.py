import os

import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from src.config import DEVICE
from src.model import create_model
from src.nih_dataset import NIHDataset


# ==========================================
# CONFIGURATION
# ==========================================

SOURCE_MODEL_PATH = "models/resnet18_finetuned.pth"

ADAPTED_MODEL_PATH = "models/resnet18_finetuned_tta.pth"

ADAPT_SPLIT = "outputs/nih_splits/adapt_images_final.csv"

BATCH_SIZE = 4
MAX_ADAPT_IMAGES = 500
LEARNING_RATE = 1e-6

# Four NIH-compatible outputs:
# Atelectasis, Cardiomegaly, Consolidation, Effusion
MODEL_INDICES = [0, 1, 2, 4]


# ==========================================
# BERNOULLI ENTROPY
# ==========================================

def bernoulli_entropy(probabilities):

    p = probabilities.clamp(
        min=1e-6,
        max=1 - 1e-6
    )

    entropy = -(
        p * torch.log(p)
        + (1 - p) * torch.log(1 - p)
    )

    return entropy.mean()


# ==========================================
# STABLE TEST-TIME ADAPTATION
# ==========================================

def stable_tta():

    print("\n====================================")
    print("FINE-TUNED RESNET18 - STABLE TTA")
    print("====================================")

    if not os.path.exists(SOURCE_MODEL_PATH):
        raise FileNotFoundError(
            f"Source model not found: {SOURCE_MODEL_PATH}"
        )

    if not os.path.exists(ADAPT_SPLIT):
        raise FileNotFoundError(
            f"Adaptation split not found: {ADAPT_SPLIT}"
        )

    # Read unlabeled NIH adaptation images.
    split_df = pd.read_csv(ADAPT_SPLIT)

    if "Image Index" not in split_df.columns:
        raise ValueError(
            "Adaptation CSV must contain 'Image Index'."
        )

    selected_images = (
        split_df["Image Index"]
        .astype(str)
        .head(MAX_ADAPT_IMAGES)
        .tolist()
    )

    if len(selected_images) != len(set(selected_images)):
        raise ValueError(
            "Duplicate image names found in adaptation split."
        )

    dataset = NIHDataset(
        image_names=set(selected_images)
    )

    if len(dataset) != len(selected_images):
        raise ValueError(
            f"Expected {len(selected_images)} adaptation images, "
            f"but loaded {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print("Adaptation Images:", len(dataset))
    print("Device:", DEVICE)
    print("Source Model:", SOURCE_MODEL_PATH)

    # Load extended ResNet18.
    model = create_model("resnet18").to(DEVICE)

    model.load_state_dict(
        torch.load(
            SOURCE_MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    # Preserve stored BatchNorm statistics.
    model.eval()

    # Freeze all parameters.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_parameters = []

    # Adapt only BatchNorm affine parameters.
    for module in model.modules():

        if isinstance(module, nn.BatchNorm2d):

            if module.weight is not None:
                module.weight.requires_grad_(True)
                trainable_parameters.append(module.weight)

            if module.bias is not None:
                module.bias.requires_grad_(True)
                trainable_parameters.append(module.bias)

    if not trainable_parameters:
        raise RuntimeError(
            "No trainable BatchNorm parameters found."
        )

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE
    )

    print("Trainable tensors:", len(trainable_parameters))
    print("Learning Rate:", LEARNING_RATE)

    # Adapt using unlabeled NIH images.
    for batch_number, (images, _) in enumerate(
        loader,
        start=1
    ):

        images = images.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)

        probabilities = torch.sigmoid(
            logits[:, MODEL_INDICES]
        )

        loss = bernoulli_entropy(probabilities)

        loss.backward()

        optimizer.step()

        if batch_number % 25 == 0:

            print(
                f"Batch {batch_number}/{len(loader)} "
                f"| Entropy Loss: {loss.item():.6f}",
                flush=True
            )

    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        ADAPTED_MODEL_PATH
    )

    print("\n====================================")
    print("STABLE TTA COMPLETED")
    print("====================================")

    print("Source Model:", SOURCE_MODEL_PATH)
    print("Adapted Model:", ADAPTED_MODEL_PATH)
    print("Adaptation Images:", len(dataset))


if __name__ == "__main__":
    stable_tta()