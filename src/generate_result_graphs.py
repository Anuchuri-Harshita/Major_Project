import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# CONFIGURATION
# ==========================================

OUTPUT_DIR = "outputs/result_graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTENDED_RESULTS = (
    "outputs/nih_extended_final_test_comparison.csv"
)

EXTENDED_DISEASE_RESULTS = (
    "outputs/nih_extended_final_test_disease_results.csv"
)

# Previous NIH results from our earlier evaluation.
# These are percentages except AUROC.
ORIGINAL_RESULTS = {
    "Original ResNet18": {
        "Accuracy": 65.99,
        "AUROC": 0.6497,
        "Sensitivity": 50.56,
        "Specificity": 68.03
    },
    "Original ResNet18 + TTA": {
        "Accuracy": 66.90,
        "AUROC": 0.6498,
        "Sensitivity": 48.81,
        "Specificity": 69.21
    }
}

MODEL_NAMES = [
    "Original\nResNet18",
    "Original\nResNet18 + TTA",
    "Extended\nResNet18",
    "Extended\nResNet18 + TTA"
]

COLORS = [
    "#90CAF9",
    "#42A5F5",
    "#1976D2",
    "#1565C0"
]


# ==========================================
# LOAD EXPERIMENTAL RESULTS
# ==========================================

def load_results():

    if not os.path.exists(EXTENDED_RESULTS):
        raise FileNotFoundError(
            f"Missing results file: {EXTENDED_RESULTS}"
        )

    if not os.path.exists(EXTENDED_DISEASE_RESULTS):
        raise FileNotFoundError(
            f"Missing disease results: {EXTENDED_DISEASE_RESULTS}"
        )

    extended_df = pd.read_csv(
        EXTENDED_RESULTS
    )

    if "Model" not in extended_df.columns:
        raise ValueError(
            "Expected a 'Model' column in extended results."
        )

    extended_df = extended_df.set_index("Model")

    before = extended_df.loc[
        "Extended ResNet18 Before TTA"
    ]

    after = extended_df.loc[
        "Extended ResNet18 After TTA"
    ]

    comparison = pd.DataFrame([
        {
            "Model": "Original ResNet18",
            **ORIGINAL_RESULTS["Original ResNet18"]
        },
        {
            "Model": "Original ResNet18 + TTA",
            **ORIGINAL_RESULTS["Original ResNet18 + TTA"]
        },
        {
            "Model": "Extended ResNet18",
            "Accuracy": before["Mean Label Accuracy"],
            "AUROC": before["Mean AUROC"],
            "Sensitivity": before["Mean Sensitivity"],
            "Specificity": before["Mean Specificity"]
        },
        {
            "Model": "Extended ResNet18 + TTA",
            "Accuracy": after["Mean Label Accuracy"],
            "AUROC": after["Mean AUROC"],
            "Sensitivity": after["Mean Sensitivity"],
            "Specificity": after["Mean Specificity"]
        }
    ])

    disease_df = pd.read_csv(
        EXTENDED_DISEASE_RESULTS
    )

    return comparison, disease_df


# ==========================================
# GRAPH 1: CHEXPERT MODEL COMPARISON
# ==========================================

def graph_chexpert():

    models = [
        "ResNet18\nBaseline",
        "DenseNet121\nBaseline",
        "ResNet18\nImproved",
        "DenseNet121\nImproved",
        "ResNet18\nExtended"
    ]

    auroc = [
        0.5491,
        0.6379,
        0.8061,
        0.7749,
        0.8100
    ]

    plt.figure(figsize=(11, 5))

    bars = plt.bar(
        models,
        auroc,
        color=[
            "#BBDEFB",
            "#90CAF9",
            "#42A5F5",
            "#64B5F6",
            "#1565C0"
        ]
    )

    plt.title(
        "CheXpert Validation: Model Comparison",
        fontsize=14,
        fontweight="bold"
    )

    plt.ylabel("Mean AUROC")
    plt.ylim(0, 1)

    for bar, value in zip(bars, auroc):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            fontsize=10
        )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/chexpert_extended_model_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# GRAPH 2: NIH ACCURACY COMPARISON
# ==========================================

def graph_accuracy(comparison):

    values = comparison["Accuracy"].tolist()

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        MODEL_NAMES,
        values,
        color=COLORS
    )

    plt.title(
        "NIH Evaluation: Model Accuracy Comparison",
        fontsize=14,
        fontweight="bold"
    )

    plt.ylabel("Mean Label Accuracy (%)")
    plt.ylim(0, 100)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=11
        )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/nih_extended_accuracy_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# GRAPH 3: NIH AUROC COMPARISON
# ==========================================

def graph_auroc(comparison):

    values = comparison["AUROC"].tolist()

    plt.figure(figsize=(11, 6))

    bars = plt.bar(
        MODEL_NAMES,
        values,
        color=COLORS
    )

    plt.title(
        "NIH Evaluation: Mean AUROC Comparison",
        fontsize=14,
        fontweight="bold"
    )

    plt.ylabel("Mean AUROC")
    plt.ylim(0, 1)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.4f}",
            ha="center",
            fontsize=11
        )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/nih_extended_auroc_comparison.png",
        dpi=300
    )

    plt.close()


# ==========================================
# GRAPH 4: SENSITIVITY AND SPECIFICITY
# ==========================================

def graph_sensitivity_specificity(comparison):

    sensitivity = comparison["Sensitivity"].to_numpy()
    specificity = comparison["Specificity"].to_numpy()

    x = np.arange(len(MODEL_NAMES))
    width = 0.35

    plt.figure(figsize=(12, 6))

    bars1 = plt.bar(
        x - width / 2,
        sensitivity,
        width,
        label="Sensitivity",
        color="#42A5F5"
    )

    bars2 = plt.bar(
        x + width / 2,
        specificity,
        width,
        label="Specificity",
        color="#1565C0"
    )

    plt.xticks(
        x,
        MODEL_NAMES
    )

    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)

    plt.title(
        "NIH Evaluation: Sensitivity vs Specificity",
        fontsize=14,
        fontweight="bold"
    )

    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1,
                f"{value:.2f}%",
                ha="center",
                fontsize=9
            )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/nih_extended_sensitivity_specificity.png",
        dpi=300
    )

    plt.close()


# ==========================================
# GRAPH 5: DISEASE-WISE AUROC
# ==========================================

def graph_disease_auroc(disease_df):

    diseases = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Effusion"
    ]

    pivot = disease_df.pivot(
        index="Disease",
        columns="Model",
        values="AUROC"
    )

    pivot = pivot.reindex(diseases)

    before = pivot[
        "Extended ResNet18 Before TTA"
    ].to_numpy()

    after = pivot[
        "Extended ResNet18 After TTA"
    ].to_numpy()

    x = np.arange(len(diseases))
    width = 0.35

    plt.figure(figsize=(11, 6))

    bars1 = plt.bar(
        x - width / 2,
        before,
        width,
        label="Before TTA",
        color="#42A5F5"
    )

    bars2 = plt.bar(
        x + width / 2,
        after,
        width,
        label="After TTA",
        color="#1565C0"
    )

    plt.xticks(x, diseases)

    plt.ylabel("AUROC")
    plt.ylim(0, 1)

    plt.title(
        "Extended ResNet18: Disease-wise AUROC Before vs After TTA",
        fontsize=13,
        fontweight="bold"
    )

    plt.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            value = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.4f}",
                ha="center",
                fontsize=9
            )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/nih_extended_disease_auroc.png",
        dpi=300
    )

    plt.close()


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n====================================")
    print("GENERATING UPDATED PROJECT GRAPHS")
    print("====================================")

    comparison, disease_df = load_results()

    graph_chexpert()
    print("1. CheXpert model comparison completed.")

    graph_accuracy(comparison)
    print("2. NIH accuracy comparison completed.")

    graph_auroc(comparison)
    print("3. NIH AUROC comparison completed.")

    graph_sensitivity_specificity(comparison)
    print("4. Sensitivity and specificity graph completed.")

    graph_disease_auroc(disease_df)
    print("5. Disease-wise AUROC graph completed.")

    print("\n====================================")
    print("ALL RESULT GRAPHS GENERATED")
    print("====================================")

    print("Saved in:", OUTPUT_DIR)

    print(
        "\nNote: NIH graphs use a previously evaluated "
        "2,000-image set, not a fresh independent test."
    )


if __name__ == "__main__":
    main()