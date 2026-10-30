import os
import random

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from src.dataset import CheXpertDataset
from src.model import create_model
from src.config import DEVICE, CLASS_NAMES


# ==========================================
# CONFIGURATION
# ==========================================

SEED = 42

TRAIN_SAMPLES = 30000
EPOCHS = 5
BATCH_SIZE = 4

OLD_MODEL_PATH = "models/resnet18_extended.pth"
NEW_MODEL_PATH = "models/resnet18_finetuned.pth"

EARLY_STOPPING_PATIENCE = 3


# ==========================================
# REPRODUCIBILITY
# ==========================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==========================================
# PATIENT-BASED IMAGE SELECTION
# ==========================================

def get_patient_subset(dataset, max_images):

    paths = dataset.data["Path"].astype(str)

    patient_ids = paths.str.extract(
        r"(patient\d+)",
        expand=False
    )

    if patient_ids.isna().any():
        raise ValueError(
            "Could not extract patient IDs."
        )

    unique_patients = patient_ids.unique().tolist()

    rng = np.random.default_rng(SEED)
    rng.shuffle(unique_patients)

    patient_array = patient_ids.to_numpy()

    selected_indices = []

    for patient in unique_patients:

        patient_indices = np.flatnonzero(
            patient_array == patient
        ).tolist()

        if (
            len(selected_indices) + len(patient_indices)
            > max_images
        ):
            continue

        selected_indices.extend(patient_indices)

        if len(selected_indices) >= max_images:
            break

    if not selected_indices:
        raise ValueError(
            "No training images selected."
        )

    return (
        Subset(dataset, selected_indices),
        selected_indices
    )


# ==========================================
# CLASS IMBALANCE
# ==========================================

def calculate_positive_weights(dataset, indices):

    labels = dataset.data.iloc[indices][
        CLASS_NAMES
    ].to_numpy(dtype=np.float32)

    labels = np.where(
        np.isfinite(labels) & (labels > 0),
        1.0,
        0.0
    )

    positives = labels.sum(axis=0)
    negatives = len(labels) - positives

    if np.any(positives == 0):
        raise ValueError(
            "Some classes have no positive samples."
        )

    weights = negatives / positives

    # Prevent extremely large positive weights.
    weights = np.clip(
        weights,
        1.0,
        10.0
    )

    print("\nClass Positive Weights:")

    for disease, weight in zip(
        CLASS_NAMES,
        weights
    ):
        print(
            f"{disease}: {weight:.2f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# ==========================================
# PREPARE MODEL
# ==========================================

def prepare_model():

    if not os.path.exists(OLD_MODEL_PATH):
        raise FileNotFoundError(
            f"Checkpoint missing: {OLD_MODEL_PATH}"
        )

    model = create_model("resnet18").to(DEVICE)

    model.load_state_dict(
        torch.load(
            OLD_MODEL_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    print(
        "\nLoaded existing checkpoint:",
        OLD_MODEL_PATH
    )

    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer3.parameters():
        parameter.requires_grad = True

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    return model


# ==========================================
# TRAINING MODE
# ==========================================

def set_training_mode(model):

    model.eval()

    model.layer3.train()
    model.layer4.train()
    model.fc.train()

    # Freeze BatchNorm running statistics.
    for module in model.modules():

        if isinstance(
            module,
            nn.modules.batchnorm._BatchNorm
        ):
            module.eval()


# ==========================================
# VALIDATION
# ==========================================

def validate_model(model, loader):

    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits
            )

            all_labels.append(
                labels.numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    y_true = np.concatenate(
        all_labels,
        axis=0
    )

    y_prob = np.concatenate(
        all_probabilities,
        axis=0
    )

    auc_scores = []

    for index in range(len(CLASS_NAMES)):

        if len(np.unique(y_true[:, index])) == 2:

            auc_scores.append(
                roc_auc_score(
                    y_true[:, index],
                    y_prob[:, index]
                )
            )

    if not auc_scores:
        raise ValueError(
            "No evaluable validation classes."
        )

    return float(np.mean(auc_scores))


# ==========================================
# TRAINING
# ==========================================

def train_model():

    set_seed()

    print("\n====================================")
    print("RESNET18 IMPROVED FINE-TUNING")
    print("====================================")

    print("Device:", DEVICE)

    train_dataset = CheXpertDataset(
        csv_file="datasets/chexpert/train.csv",
        image_root="datasets/chexpert",
        train=True
    )

    valid_dataset = CheXpertDataset(
        csv_file="datasets/chexpert/valid.csv",
        image_root="datasets/chexpert",
        train=False
    )

    train_subset, selected_indices = get_patient_subset(
        train_dataset,
        TRAIN_SAMPLES
    )

    print(
        "Training Images:",
        len(train_subset)
    )

    print(
        "Validation Images:",
        len(valid_dataset)
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    positive_weights = calculate_positive_weights(
        train_dataset,
        selected_indices
    ).to(DEVICE)

    model = prepare_model()

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weights
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.layer3.parameters(),
                "lr": 0.000001
            },
            {
                "params": model.layer4.parameters(),
                "lr": 0.000005
            },
            {
                "params": model.fc.parameters(),
                "lr": 0.00001
            }
        ],
        weight_decay=0.0001
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    initial_auc = validate_model(
        model,
        valid_loader
    )

    print(
        f"\nStarting Validation AUROC: "
        f"{initial_auc:.4f}"
    )

    best_auc = initial_auc

    # Save the starting model so the output checkpoint
    # always exists, even if fine-tuning does not improve it.
    torch.save(
        model.state_dict(),
        NEW_MODEL_PATH
    )

    epochs_without_improvement = 0

    for epoch in range(EPOCHS):

        set_training_mode(model)

        running_loss = 0.0

        for batch_number, (images, labels) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                ],
                max_norm=1.0
            )

            optimizer.step()

            running_loss += loss.item()

            if batch_number % 250 == 0:

                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"Batch {batch_number}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True
                )

        average_loss = (
            running_loss / len(train_loader)
        )

        validation_auc = validate_model(
            model,
            valid_loader
        )

        scheduler.step(
            validation_auc
        )

        print("\n------------------------------------")
        print(
            f"Epoch: {epoch + 1}/{EPOCHS}"
        )
        print(
            f"Training Loss: {average_loss:.4f}"
        )
        print(
            f"Validation Mean AUROC: "
            f"{validation_auc:.4f}"
        )
        print("------------------------------------")

        if validation_auc > best_auc:

            best_auc = validation_auc

            torch.save(
                model.state_dict(),
                NEW_MODEL_PATH
            )

            epochs_without_improvement = 0

            print(
                "Improved checkpoint saved:",
                NEW_MODEL_PATH
            )

        else:

            epochs_without_improvement += 1

            print(
                "No validation AUROC improvement."
            )

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping activated."
            )

            break

    print("\n====================================")
    print("TRAINING COMPLETED")
    print("====================================")

    print(
        f"Starting AUROC: {initial_auc:.4f}"
    )

    print(
        f"Best AUROC: {best_auc:.4f}"
    )

    print(
        "Saved checkpoint:",
        NEW_MODEL_PATH
    )


if __name__ == "__main__":
    train_model()