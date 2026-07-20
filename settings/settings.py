"""Shared settings for the Zoodle project."""

from pathlib import Path


# ============================================================
# 1. SÖKVÄGAR
# ============================================================
# Rådata sparas separat så att originalfilerna inte förändras senare i flödet.
RAW_DATA_DIR = Path("data") / "raw"
PROCESSED_DATA_DIR = Path("data") / "processed"
EDA_OUTPUT_DIR = Path("output") / "eda"
PCA_OUTPUT_DIR = Path("output") / "pca"
UMAP_OUTPUT_DIR = Path("output") / "umap"
TRAINING_OUTPUT_DIR = Path("output") / "training"
MODEL_OUTPUT_DIR = Path("artifacts") / "training"


# ============================================================
# 2. DATASET
# ============================================================
# Quick, Draw!-filerna innehåller 28 x 28 pixlar stora gråskalebilder i NumPy-format.
DATASET_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

# Djurklasserna används i samma ordning genom hela projektet.
ANIMAL_CLASSES = [
    "bat",
    "bear",
    "camel",
    "cat",
    "cow",
    "dog",
    "elephant",
    "giraffe",
    "hedgehog",
    "horse",
    "kangaroo",
    "lion",
    "monkey",
    "mouse",
    "panda",
    "pig",
    "rabbit",
    "raccoon",
    "rhinoceros",
    "sheep",
    "squirrel",
    "tiger",
    "zebra",
    "bird",
    "duck",
    "flamingo",
    "owl",
    "parrot",
    "penguin",
    "swan",
    "crab",
    "dolphin",
    "fish",
    "lobster",
    "octopus",
    "sea turtle",
    "shark",
    "whale",
    "ant",
    "bee",
    "butterfly",
    "mosquito",
    "scorpion",
    "snail",
    "spider",
    "crocodile",
    "frog",
    "snake",
]

# Djurgrupperna används för att skapa mer lättlästa visualiseringar.
ANIMAL_GROUPS = {
    "Däggdjur": [
        "bat",
        "bear",
        "camel",
        "cat",
        "cow",
        "dog",
        "elephant",
        "giraffe",
        "hedgehog",
        "horse",
        "kangaroo",
        "lion",
        "monkey",
        "mouse",
        "panda",
        "pig",
        "rabbit",
        "raccoon",
        "rhinoceros",
        "sheep",
        "squirrel",
        "tiger",
        "zebra",
    ],
    "Fåglar": [
        "bird",
        "duck",
        "flamingo",
        "owl",
        "parrot",
        "penguin",
        "swan",
    ],
    "Vattendjur": [
        "crab",
        "dolphin",
        "fish",
        "lobster",
        "octopus",
        "sea turtle",
        "shark",
        "whale",
    ],
    "Insekter och små ryggradslösa djur": [
        "ant",
        "bee",
        "butterfly",
        "mosquito",
        "scorpion",
        "snail",
        "spider",
    ],
    "Reptiler och groddjur": [
        "crocodile",
        "frog",
        "snake",
    ],
}


# ============================================================
# 3. DATAFÖRBEREDELSE
# ============================================================
# Samma seed används för urval och uppdelning så att resultatet kan återskapas.
IMAGE_SIZE = 28
SAMPLES_PER_CLASS = 5000
RANDOM_STATE = 42

# Datasetet delas upp i träning, validering och test.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# 4. PCA OCH UMAP
# ============================================================
# Ett balanserat urval håller analysen hanterbar utan att någon klass dominerar.
PCA_SAMPLES_PER_CLASS = 500
UMAP_SAMPLES_PER_CLASS = 250
PCA_VARIANCE = 0.95
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


# ============================================================
# 5. MODELLTRÄNING
# ============================================================
# Träningsinställningarna kan ändras mellan körningar för att jämföra resultat.
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5
DROPOUT_RATE = 0.25
USE_DATA_AUGMENTATION = True
AUGMENTATION_ROTATION = 0.08
AUGMENTATION_TRANSLATION = 0.1
AUGMENTATION_ZOOM = 0.1
