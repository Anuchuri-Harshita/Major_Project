import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.dataset import CheXpertDataset
from src.model import create_model

from src.config import (
    DEVICE,
    CLASS_NAMES,
    RESNET_IMPROVED_PATH
)


# ----------------------------------
# Grad-CAM Generation
# ----------------------------------

def generate_gradcam(image_index=0, disease_name="Cardiomegaly"):

    print("\n====================================")
    print("CHEST X-RAY GRAD-CAM GENERATION")
    print("====================================")

    # Validate disease name
    if disease_name not in CLASS_NAMES:
        raise ValueError(
            f"Invalid disease: {disease_name}\n"
            f"Choose from: {CLASS_NAMES}"
        )

    disease_index = CLASS_NAMES.index(disease_name)

    # ----------------------------------
    # Load CheXpert Validation Dataset
    # ----------------------------------

    dataset = CheXpertDataset(
        csv_file="datasets/chexpert/valid.csv",
        image_root="datasets/chexpert"
    )

    if image_index < 0 or image_index >= len(dataset):
        raise IndexError(
            f"Image index must be between 0 and {len(dataset)-1}"
        )

    image_tensor, labels = dataset[image_index]

    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    actual_label = int(labels[disease_index].item())

    # ----------------------------------
    # Load Improved ResNet18
    # ----------------------------------

    model = create_model("resnet18").to(DEVICE)

    model.load_state_dict(
        torch.load(
            RESNET_IMPROVED_PATH,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    # ----------------------------------
    # Get Model Prediction
    # ----------------------------------

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.sigmoid(outputs)

        probability = probabilities[0, disease_index].item()

    predicted_label = 1 if probability >= 0.5 else 0

    # ----------------------------------
    # Prepare Original Image
    # ----------------------------------

    # Reverse ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    original_image = image_tensor.permute(1, 2, 0).cpu().numpy()

    original_image = original_image * std + mean

    original_image = np.clip(original_image, 0, 1)

    original_image = original_image.astype(np.float32)

    # ----------------------------------
    # Generate Grad-CAM
    # ----------------------------------

    # Final convolutional block of ResNet18
    target_layers = [model.layer4[-1]]

    # Explain the selected disease logit.
    # Do not use softmax because this is
    # a multi-label classification problem.
    targets = [
        ClassifierOutputTarget(disease_index)
    ]

    with GradCAM(
        model=model,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=targets
        )[0]

    # ----------------------------------
    # Create Heatmap and Overlay
    # ----------------------------------

    heatmap = plt.get_cmap("jet")(grayscale_cam)[..., :3]

    overlay = show_cam_on_image(
        original_image,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.5
    )

    # ----------------------------------
    # Save Images
    # ----------------------------------

    output_dir = "outputs/gradcam"

    os.makedirs(output_dir, exist_ok=True)

    file_prefix = (
        f"image_{image_index}_"
        f"{disease_name.lower().replace(' ', '_')}"
    )

    original_path = os.path.join(
        output_dir,
        f"{file_prefix}_original.png"
    )

    heatmap_path = os.path.join(
        output_dir,
        f"{file_prefix}_heatmap.png"
    )

    overlay_path = os.path.join(
        output_dir,
        f"{file_prefix}_overlay.png"
    )

    plt.imsave(
        original_path,
        original_image
    )

    plt.imsave(
        heatmap_path,
        heatmap
    )

    Image.fromarray(overlay).save(overlay_path)

    # ----------------------------------
    # Create Comparison Figure
    # ----------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5)
    )

    axes[0].imshow(original_image)
    axes[0].set_title("Original Chest X-ray")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis("off")

    actual_text = (
        "Positive" if actual_label == 1 else "Negative"
    )

    predicted_text = (
        "Positive" if predicted_label == 1 else "Negative"
    )

    fig.suptitle(
        f"ResNet18 Improved | {disease_name}\n"
        f"Actual: {actual_text} | "
        f"Predicted: {predicted_text} | "
        f"Probability: {probability * 100:.2f}%",
        fontsize=13
    )

    plt.tight_layout()

    comparison_path = os.path.join(
        output_dir,
        f"{file_prefix}_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ----------------------------------
    # Display Results
    # ----------------------------------

    print("\nModel: ResNet18 Improved")

    print("Dataset: CheXpert Validation")

    print("Image Index:", image_index)

    print("Disease:", disease_name)

    print("Actual Label:", actual_text)

    print("Predicted Label:", predicted_text)

    print(
        f"Predicted Probability: {probability * 100:.2f}%"
    )

    print("\nGrad-CAM generated successfully!")

    print("\nSaved files:")

    print(original_path)
    print(heatmap_path)
    print(overlay_path)
    print(comparison_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--index",
        type=int,
        default=0
    )

    parser.add_argument(
        "--disease",
        type=str,
        default="Cardiomegaly"
    )

    args = parser.parse_args()

    generate_gradcam(
        image_index=args.index,
        disease_name=args.disease
    )