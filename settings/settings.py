"""Shared settings for the Zoodle project."""

from pathlib import Path


# ============================================================
# 1. SÖKVÄGAR
# ============================================================
# Rådata sparas separat så att originalfilerna inte förändras senare i flödet.
RAW_DATA_DIR = Path("data") / "raw"
PROCESSED_DATA_DIR = Path("data") / "processed"


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
