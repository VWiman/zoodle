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
EVALUATION_OUTPUT_DIR = Path("output") / "evaluation"
CNN_EVALUATION_OUTPUT_DIR = EVALUATION_OUTPUT_DIR / "cnn"
KNN_TRAINING_OUTPUT_DIR = Path("output") / "knn"
KNN_MODEL_OUTPUT_DIR = Path("artifacts") / "knn"
KNN_EVALUATION_OUTPUT_DIR = EVALUATION_OUTPUT_DIR / "knn"


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

# Djurklassernas svenska namn visas i Streamlit-appen.
ANIMAL_NAMES_SV = {
    "bat": "fladdermus",
    "bear": "björn",
    "camel": "kamel",
    "cat": "katt",
    "cow": "ko",
    "dog": "hund",
    "elephant": "elefant",
    "giraffe": "giraff",
    "hedgehog": "igelkott",
    "horse": "häst",
    "kangaroo": "känguru",
    "lion": "lejon",
    "monkey": "apa",
    "mouse": "mus",
    "panda": "panda",
    "pig": "gris",
    "rabbit": "kanin",
    "raccoon": "tvättbjörn",
    "rhinoceros": "noshörning",
    "sheep": "får",
    "squirrel": "ekorre",
    "tiger": "tiger",
    "zebra": "zebra",
    "bird": "fågel",
    "duck": "anka",
    "flamingo": "flamingo",
    "owl": "uggla",
    "parrot": "papegoja",
    "penguin": "pingvin",
    "swan": "svan",
    "crab": "krabba",
    "dolphin": "delfin",
    "fish": "fisk",
    "lobster": "hummer",
    "octopus": "bläckfisk",
    "sea turtle": "havssköldpadda",
    "shark": "haj",
    "whale": "val",
    "ant": "myra",
    "bee": "bi",
    "butterfly": "fjäril",
    "mosquito": "mygga",
    "scorpion": "skorpion",
    "snail": "snigel",
    "spider": "spindel",
    "crocodile": "krokodil",
    "frog": "groda",
    "snake": "orm",
}

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
SAMPLES_PER_CLASS = 20000
RANDOM_STATE = 42

# Datasetet delas upp i träning, validering och test.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# 4. PCA OCH UMAP
# ============================================================
# Ett balanserat urval håller analysen hanterbar utan att någon klass dominerar.
PCA_SAMPLES_PER_CLASS = 1000
UMAP_SAMPLES_PER_CLASS = 500
PCA_VARIANCE = 0.95
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1


# ============================================================
# 5. CNN-MODELL OCH TRÄNING
# ============================================================
# CNN-inställningarna kan ändras mellan körningar för att jämföra resultat.
CNN_BATCH_SIZE = 128
CNN_EPOCHS = 45
CNN_LEARNING_RATE = 1e-4
# Eager mode används för att undvika felaktiga graph-beräkningar på Metal.
CNN_RUN_EAGERLY = True
CNN_USE_EARLY_STOPPING = True
CNN_EARLY_STOPPING_PATIENCE = 5
CNN_USE_REDUCE_LR_ON_PLATEAU = True
CNN_REDUCE_LR_PATIENCE = 2
CNN_REDUCE_LR_FACTOR = 0.5
CNN_MIN_LEARNING_RATE = 1e-6
CNN_DROPOUT_RATE = 0.3
CNN_USE_DATA_AUGMENTATION = True
CNN_AUGMENTATION_ROTATION = 0.08
CNN_AUGMENTATION_TRANSLATION = 0.1
CNN_AUGMENTATION_ZOOM = 0.1


# ============================================================
# 6. PCA + KNN-MODELL
# ============================================================
# KNN använder hela den split som är aktuell för träning, validering eller test.
KNN_DATA_FRACTION = 1.00
KNN_PCA_VARIANCE = 0.95
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"
KNN_METRIC = "euclidean"
KNN_ALGORITHM = "brute"
KNN_N_JOBS = -1
KNN_PREDICTION_BATCH_SIZE = 128


# ============================================================
# 7. MODELLUTVÄRDERING
# ============================================================
# De vanligaste förväxlingarna och tydliga felexempel lyfts fram i resultatet.
TOP_CONFUSIONS = 15
MISCLASSIFIED_EXAMPLES = 16
