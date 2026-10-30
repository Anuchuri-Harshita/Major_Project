import os
import numpy as np
import pandas as pd


# ==========================================
# CONFIGURATION
# ==========================================

NIH_ROOT = "datasets/nih"

CSV_PATH = os.path.join(
    NIH_ROOT,
    "Data_Entry_2017_v2020.csv"
)

TEST_LIST = os.path.join(
    NIH_ROOT,
    "test_list.txt"
)

OUTPUT_DIR = "outputs/nih_splits"

SEED = 142
FRESH_TEST_SIZE = 2000

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "fresh_test_images.csv"
)


# ==========================================
# READ IMAGE LIST
# ==========================================

def read_image_list(path):

    with open(path, "r", encoding="utf-8") as file:

        return {
            line.strip()
            for line in file
            if line.strip()
        }


# ==========================================
# SELECT COMPLETE PATIENT GROUPS
# ==========================================

def select_images_by_patient(
    data,
    max_images,
    seed
):

    rng = np.random.default_rng(seed)

    patient_groups = [
        group["Image Index"].tolist()
        for _, group in data.groupby(
            "Patient ID",
            sort=True
        )
    ]

    rng.shuffle(patient_groups)

    selected = []

    for patient_images in patient_groups:

        if (
            len(selected) + len(patient_images)
            > max_images
        ):
            continue

        selected.extend(patient_images)

        if len(selected) == max_images:
            break

    return selected


# ==========================================
# PREPARE FRESH TEST SPLIT
# ==========================================

def prepare_final_splits():

    print("\n====================================")
    print("PREPARING FRESH NIH TEST SPLIT")
    print("====================================")

    if os.path.exists(OUTPUT_FILE):

        raise FileExistsError(
            f"{OUTPUT_FILE} already exists. "
            "Refusing to overwrite an existing test split."
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    data = pd.read_csv(CSV_PATH)

    test_names = read_image_list(TEST_LIST)

    test_data = data[
        data["Image Index"].isin(test_names)
    ].copy()

    # Collect patients from all previous split CSVs.
    previously_used_images = set()

    split_files = [
        "adapt_images.csv",
        "test_images.csv",
        "adapt_images_final.csv",
        "dev_images.csv",
        "final_test_images.csv"
    ]

    for filename in split_files:

        path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        if not os.path.exists(path):
            continue

        previous = pd.read_csv(path)

        if "Image Index" not in previous.columns:
            raise ValueError(
                f"Missing Image Index column in {path}"
            )

        previously_used_images.update(
            previous["Image Index"]
            .astype(str)
            .tolist()
        )

        print(
            "Excluded previous split:",
            filename
        )

    previously_used_patients = set(
        data.loc[
            data["Image Index"].isin(
                previously_used_images
            ),
            "Patient ID"
        ]
    )

    print(
        "\nPreviously Used Patients:",
        len(previously_used_patients)
    )

    # Exclude all previously used patients.
    fresh_candidates = test_data[
        ~test_data["Patient ID"].isin(
            previously_used_patients
        )
    ].copy()

    print(
        "Available Candidate Images:",
        len(fresh_candidates)
    )

    fresh_images = select_images_by_patient(
        fresh_candidates,
        FRESH_TEST_SIZE,
        SEED
    )

    if not fresh_images:
        raise ValueError(
            "No eligible fresh test images found."
        )

    fresh_patients = set(
        data.loc[
            data["Image Index"].isin(
                fresh_images
            ),
            "Patient ID"
        ]
    )

    assert not (
        fresh_patients & previously_used_patients
    ), "Patient leakage detected!"

    if len(fresh_images) != FRESH_TEST_SIZE:

        print(
            f"\nWarning: Requested {FRESH_TEST_SIZE} "
            f"images, selected {len(fresh_images)}."
        )

    pd.DataFrame({
        "Image Index": fresh_images
    }).to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n====================================")
    print("FRESH NIH TEST SPLIT CREATED")
    print("====================================")

    print(
        "Fresh Test Images:",
        len(fresh_images)
    )

    print(
        "Fresh Test Patients:",
        len(fresh_patients)
    )

    print(
        "Overlap With Previous Patients:",
        len(
            fresh_patients
            & previously_used_patients
        )
    )

    print(
        "\nSaved:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    prepare_final_splits()