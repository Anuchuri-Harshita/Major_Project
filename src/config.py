import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

IMAGE_SIZE = 224

BATCH_SIZE = 4

EPOCHS = 3

LEARNING_RATE = 0.00001

TRAIN_SAMPLES = 15000

CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion"
]

NUM_CLASSES = len(CLASS_NAMES)

# Previous baseline models
RESNET_MODEL_PATH = "models/resnet18_baseline.pth"
DENSENET_MODEL_PATH = "models/densenet121_baseline.pth"

# Improved models
RESNET_IMPROVED_PATH = "models/resnet18_improved.pth"
DENSENET_IMPROVED_PATH = "models/densenet121_improved.pth"

MODEL_PATH = RESNET_MODEL_PATH