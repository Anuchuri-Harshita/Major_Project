import os

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.config import DEVICE
from src.model import create_model
from src.nih_dataset import NIHDataset


# ==========================================
# CONFIGURATION
# ==========================================

BATCH_SIZE = 4

NIH_CSV = "datasets/nih/Data_Entry_2017_v2020.csv"

SPLIT_DIR = "outputs/nih_splits"

SPLIT_FILE = os.path.join(
    SPLIT_DIR,
    "fresh_test_images.csv"
)

MODEL_INDICES = [0, 1, 2, 4]

DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Effusion"
]

MODELS = {
    "Fine-tuned ResNet18 Before TTA":
        "models/resnet18_finetuned.pth",

    "Fine-tuned ResNet18 After TTA":
        "models/resnet18_finetuned_tta.pth"
}

SUMMARY_OUTPUT = (
    "outputs/nih_finetuned_fresh_test_comparison.csv"
)

DISEASE_OUTPUT = (
    "outputs/nih_finetuned_fresh_test_disease_results.csv"
)

THRESHOLD = 0.5

PREVIOUS_SPLITS = [
    "adapt_images.csv",
    "test_images.csv",
    "adapt_images_final.csv",
    "dev_images.csv",
    "final_test_images.csv"
]


# ==========================================
# CHECK PATIENT AND IMAGE OVERLAP
# ==========================================

def check_split_overlap():

    print("\nChecking NIH split overlap...")

    if not os.path.exists(NIH_CSV):
        raise FileNotFoundError(NIH_CSV)

    if not os.path.exists(SPLIT_FILE):
        raise FileNotFoundError(SPLIT_FILE)

    metadata = pd.read_csv(NIH_CSV)

    required_columns = {
        "Image Index",
        "Patient ID"
    }

    if not required_columns.issubset(metadata.columns):
        raise ValueError(
            "NIH metadata is missing required columns."
        )

    if metadata["Image Index"].duplicated().any():
        raise ValueError(
            "Duplicate image names in NIH metadata."
        )

    patient_lookup = metadata.set_index(
        "Image Index"
    )["Patient ID"]

    fresh_df = pd.read_csv(SPLIT_FILE)

    if "Image Index" not in fresh_df.columns:
        raise ValueError(
            "Fresh test split has no Image Index column."
        )

    fresh_names = fresh_df[
        "Image Index"
    ].astype(str)

    if fresh_names.duplicated().any():
        raise ValueError(
            "Duplicate images in fresh test split."
        )

    fresh_images = set(fresh_names)

    unknown_images = (
        fresh_images - set(patient_lookup.index)
    )

    if unknown_images:
        raise ValueError(
            "Some fresh test images are missing "
            "from NIH metadata."
        )

    fresh_patients = set(
        patient_lookup.loc[
            list(fresh_images)
        ]
    )

    previous_images = set()

    for filename in PREVIOUS_SPLITS:

        path = os.path.join(
            SPLIT_DIR,
            filename
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Previous split missing: {path}"
            )

        previous_df = pd.read_csv(path)

        if "Image Index" not in previous_df.columns:
            raise ValueError(
                f"Image Index missing in {path}"
            )

        previous_images.update(
            previous_df["Image Index"]
            .astype(str)
            .tolist()
        )

    unknown_previous = (
        previous_images - set(patient_lookup.index)
    )

    if unknown_previous:
        raise ValueError(
            "Some previous images are missing "
            "from NIH metadata."
        )

    previous_patients = set(
        patient_lookup.loc[
            list(previous_images)
        ]
    )

    image_overlap = (
        fresh_images & previous_images
    )

    patient_overlap = (
        fresh_patients & previous_patients
    )

    print(
        "Fresh Test Images:",
        len(fresh_images)
    )

    print(
        "Fresh Test Patients:",
        len(fresh_patients)
    )

    print(
        "Previously Used Patients:",
        len(previous_patients)
    )

    print(
        "Image Overlap:",
        len(image_overlap)
    )

    print(
        "Patient Overlap:",
        len(patient_overlap)
    )

    if image_overlap or patient_overlap:
        raise ValueError(
            "Data leakage detected between fresh "
            "test and previously used NIH images."
        )

    # Confirm that fresh images belong to the
    # official NIH test partition.
    official_test_path = (
        "datasets/nih/test_list.txt"
    )

    with open(
        official_test_path,
        "r",
        encoding="utf-8"
    ) as file:

        official_test_images = {
            line.strip()
            for line in file
            if line.strip()
        }

    if not fresh_images.issubset(
        official_test_images
    ):
        raise ValueError(
            "Fresh test contains images outside "
            "the official NIH test partition."
        )

    print(
        "Patient separation verified successfully."
    )

    return fresh_images


# ==========================================
# LOAD MODEL AND PREDICTIONS
# ==========================================

def get_predictions(model_path, loader):

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = create_model(
        "resnet18"
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for batch_number, (images, labels) in enumerate(
            loader,
            start=1
        ):

            images = images.to(DEVICE)

            logits = model(images)

            probabilities = torch.sigmoid(
                logits[:, MODEL_INDICES]
            )

            all_labels.append(
                labels.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

            if batch_number % 100 == 0:

                print(
                    f"Processed {batch_number}/{len(loader)} batches",
                    flush=True
                )

    y_true = np.concatenate(
        all_labels,
        axis=0
    )

    y_prob = np.concatenate(
        all_probabilities,
        axis=0
    )

    if y_true.shape != y_prob.shape:
        raise ValueError(
            "Prediction and label shapes differ."
        )

    if y_true.shape[1] != len(DISEASES):
        raise ValueError(
            "Unexpected number of disease labels."
        )

    return y_true, y_prob


# ==========================================
# CALCULATE METRICS
# ==========================================

def calculate_metrics(y_true, y_prob):

    y_pred = (
        y_prob >= THRESHOLD
    ).astype(int)

    disease_results = []

    for index, disease in enumerate(DISEASES):

        actual = y_true[:, index]
        predicted = y_pred[:, index]
        scores = y_prob[:, index]

        tp = int(np.sum(
            (actual == 1) & (predicted == 1)
        ))

        tn = int(np.sum(
            (actual == 0) & (predicted == 0)
        ))

        fp = int(np.sum(
            (actual == 0) & (predicted == 1)
        ))

        fn = int(np.sum(
            (actual == 1) & (predicted == 0)
        ))

        accuracy = (
            (tp + tn) / len(actual)
        ) * 100

        sensitivity = (
            tp / (tp + fn) * 100
            if (tp + fn) > 0
            else np.nan
        )

        specificity = (
            tn / (tn + fp) * 100
            if (tn + fp) > 0
            else np.nan
        )

        auroc = (
            roc_auc_score(actual, scores)
            if len(np.unique(actual)) == 2
            else np.nan
        )

        disease_results.append({
            "Disease": disease,
            "Accuracy": accuracy,
            "AUROC": auroc,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Actual Positive": tp + fn,
            "Predicted Positive": tp + fp,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        })

    disease_df = pd.DataFrame(
        disease_results
    )

    # Accuracy if every disease is predicted absent.
    all_negative_accuracy = (
        (y_true == 0).mean() * 100
    )

    summary = {
        "Mean Label Accuracy":
            disease_df["Accuracy"].mean(),

        "Mean AUROC":
            disease_df["AUROC"].mean(),

        "Mean Sensitivity":
            disease_df["Sensitivity"].mean(),

        "Mean Specificity":
            disease_df["Specificity"].mean(),

        "All-Negative Baseline Accuracy":
            all_negative_accuracy
    }

    return summary, disease_df


# ==========================================
# MAIN EVALUATION
# ==========================================

def main():

    print("\n====================================")
    print("FINE-TUNED RESNET18 FRESH NIH TEST")
    print("====================================")

    selected_images = check_split_overlap()

    dataset = NIHDataset(
        image_names=selected_images
    )

    if len(dataset) != len(selected_images):
        raise ValueError(
            f"Expected {len(selected_images)} images, "
            f"but loaded {len(dataset)}."
        )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    print(
        "\nEvaluation Images:",
        len(dataset)
    )

    print("Device:", DEVICE)
    print("Threshold:", THRESHOLD)

    all_summaries = []
    all_disease_results = []

    for model_name, model_path in MODELS.items():

        print("\n====================================")
        print("Evaluating:", model_name)
        print("====================================")

        y_true, y_prob = get_predictions(
            model_path,
            loader
        )

        summary, disease_df = calculate_metrics(
            y_true,
            y_prob
        )

        summary["Model"] = model_name

        all_summaries.append(summary)

        disease_df.insert(
            0,
            "Model",
            model_name
        )

        all_disease_results.append(
            disease_df
        )

    summary_df = pd.DataFrame(
        all_summaries
    ).set_index("Model")

    disease_df = pd.concat(
        all_disease_results,
        ignore_index=True
    )

    print("\n====================================")
    print("FINAL FRESH NIH TEST RESULTS")
    print("====================================")

    print(
        summary_df.round(4).to_string()
    )

    print("\nDisease-wise results:")

    print(
        disease_df[
            [
                "Model",
                "Disease",
                "Accuracy",
                "AUROC",
                "Sensitivity",
                "Specificity"
            ]
        ].round(4).to_string(index=False)
    )

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    if (
        os.path.exists(SUMMARY_OUTPUT)
        or os.path.exists(DISEASE_OUTPUT)
    ):
        print("\nExisting final test results found.")
        print("Skipping CSV saving to preserve previous results.")

    else:
        summary_df.to_csv(
            SUMMARY_OUTPUT
        )

        disease_df.to_csv(
            DISEASE_OUTPUT,
            index=False
        )

        print("\nFinal results saved successfully!")

    print("\n====================================")
    print("FINAL EVALUATION COMPLETED")
    print("====================================")

    print(
        "Summary:",
        SUMMARY_OUTPUT
    )

    print(
        "Disease-wise:",
        DISEASE_OUTPUT
    )

    print(
        "\nThis is the first evaluation on the "
        "fresh patient-disjoint NIH test set."
    )


if __name__ == "__main__":
    main()