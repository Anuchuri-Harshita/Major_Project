import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from src.config import DEVICE
from src.model import create_model


# ==========================================
# CONFIGURATION
# ==========================================

NIH_ROOT = "datasets/nih/images"

NIH_CSV = "datasets/nih/Data_Entry_2017_v2020.csv"

SPLIT_FILE = "outputs/nih_splits/dev_images.csv"

OUTPUT_DIR = "outputs/gradcam_nih_extended"

BEFORE_MODEL_PATH = "models/resnet18_extended.pth"

AFTER_MODEL_PATH = "models/resnet18_extended_tta.pth"

DISEASES = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Consolidation": 2,
    "Effusion": 4
}

IMAGE_SIZE = 224


# ==========================================
# FIND NIH IMAGE
# ==========================================

def find_image(image_name):

    for root, _, files in os.walk(NIH_ROOT):

        if image_name in files:
            return os.path.join(root, image_name)

    raise FileNotFoundError(
        f"Image not found: {image_name}"
    )


# ==========================================
# LOAD IMAGE
# ==========================================

def load_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    tensor = transform(image).unsqueeze(0)

    return image, tensor.to(DEVICE)


# ==========================================
# LOAD MODEL
# ==========================================

def load_model(model_path):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = create_model("resnet18").to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    return model


# ==========================================
# GENERATE GRAD-CAM
# ==========================================

def generate_gradcam(model, image_tensor, class_index):

    activations = []
    gradients = []

    target_layer = model.layer4[-1]

    def forward_hook(module, inputs, output):
        activations.append(output.detach())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        model.zero_grad(set_to_none=True)

        logits = model(image_tensor)

        probability = torch.sigmoid(
            logits[0, class_index]
        ).item()

        score = logits[0, class_index]

        score.backward()

        activation = activations[0]
        gradient = gradients[0]

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activation
        ).sum(dim=1, keepdim=True)

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam[0, 0].cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max > cam_min:
            cam = (
                cam - cam_min
            ) / (
                cam_max - cam_min
            )
        else:
            cam = np.zeros_like(cam)

        return cam, probability

    finally:

        forward_handle.remove()
        backward_handle.remove()


# ==========================================
# GET ACTUAL NIH LABEL
# ==========================================

def get_actual_label(image_name, disease):

    if not os.path.exists(NIH_CSV):
        return "Unknown"

    metadata = pd.read_csv(
        NIH_CSV,
        usecols=["Image Index", "Finding Labels"]
    )

    matching_rows = metadata[
        metadata["Image Index"] == image_name
    ]

    if matching_rows.empty:
        return "Unknown"

    findings = str(
        matching_rows.iloc[0]["Finding Labels"]
    ).split("|")

    return (
        "Positive"
        if disease in findings
        else "Negative"
    )


# ==========================================
# MAIN
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        default=0
    )

    parser.add_argument(
        "--disease",
        type=str,
        default="Atelectasis",
        choices=list(DISEASES.keys())
    )

    args = parser.parse_args()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_FILE}"
        )

    images_df = pd.read_csv(SPLIT_FILE)

    if args.index < 0 or args.index >= len(images_df):
        raise ValueError("Invalid image index.")

    image_name = images_df.iloc[
        args.index
    ]["Image Index"]

    image_path = find_image(image_name)

    actual_label = get_actual_label(
        image_name,
        args.disease
    )

    print("\n====================================")
    print("EXTENDED RESNET18 GRAD-CAM")
    print("====================================")

    print("Image:", image_name)
    print("Disease:", args.disease)
    print("Actual Label:", actual_label)
    print("Device:", DEVICE)

    image, image_tensor = load_image(
        image_path
    )

    class_index = DISEASES[
        args.disease
    ]

    # ======================================
    # BEFORE TTA
    # ======================================

    print("\nLoading Extended ResNet18 Before TTA...")

    before_model = load_model(
        BEFORE_MODEL_PATH
    )

    before_cam, before_probability = generate_gradcam(
        before_model,
        image_tensor,
        class_index
    )

    del before_model

    # ======================================
    # AFTER TTA
    # ======================================

    print("Loading Extended ResNet18 After TTA...")

    after_model = load_model(
        AFTER_MODEL_PATH
    )

    after_cam, after_probability = generate_gradcam(
        after_model,
        image_tensor,
        class_index
    )

    del after_model

    # ======================================
    # VISUALIZATION
    # ======================================

    image_array = np.array(image)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(image_array)

    axes[0].set_title(
        f"Original NIH Chest X-ray\n"
        f"Actual Label: {actual_label}"
    )

    axes[1].imshow(image_array)

    axes[1].imshow(
        before_cam,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1
    )

    axes[1].set_title(
        f"Before TTA\n"
        f"Probability: {before_probability:.4f}"
    )

    axes[2].imshow(image_array)

    axes[2].imshow(
        after_cam,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        f"After Stable TTA\n"
        f"Probability: {after_probability:.4f}"
    )

    for axis in axes:
        axis.axis("off")

    plt.suptitle(
        f"{args.disease} - Grad-CAM Comparison",
        fontsize=15
    )

    plt.tight_layout()

    safe_disease = args.disease.replace(
        " ",
        "_"
    )

    filename = (
        f"gradcam_extended_{args.index}_"
        f"{safe_disease}.png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\n====================================")
    print("GRAD-CAM COMPLETED")
    print("====================================")

    print(
        "Before TTA Probability:",
        round(before_probability, 4)
    )

    print(
        "After TTA Probability:",
        round(after_probability, 4)
    )

    print("Saved Image:", output_path)


if __name__ == "__main__":
    main()