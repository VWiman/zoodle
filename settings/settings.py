"""Shared settings for the Zoodle project."""

from pathlib import Path


# ============================================================
# 1. SÖKVÄGAR
# ============================================================
# Rådata sparas separat så att originalfilerna inte förändras senare i flödet.
RAW_DATA_DIR = Path("data") / "raw"


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
