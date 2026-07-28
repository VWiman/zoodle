"""Build the PCA and KNN baseline model used by Zoodle."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from settings import (
    IMAGE_SIZE,
    KNN_ALGORITHM,
    KNN_METRIC,
    KNN_N_JOBS,
    KNN_N_NEIGHBORS,
    KNN_PCA_VARIANCE,
    KNN_WEIGHTS,
    RANDOM_STATE,
)


# ============================================================
# 1. FÖRBERED BILDER FÖR KNN
# ============================================================
# KNN använder platta bilder. Pixelvärdena normaliseras till intervallet 0–1
# innan PCA beräknar den reducerade representationen.
def prepare_knn_images(images: np.ndarray) -> np.ndarray:
    expected_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)

    if images.ndim != 4 or images.shape[1:] != expected_shape:
        raise ValueError(
            f"Fel bildform: {images.shape}. Förväntad form är "
            f"(antal bilder, {IMAGE_SIZE}, {IMAGE_SIZE}, 1)."
        )

    flattened_images = images.reshape(len(images), IMAGE_SIZE * IMAGE_SIZE)
    prepared_images = flattened_images.astype(np.float32)
    prepared_images /= 255.0

    return prepared_images


# ============================================================
# 2. BYGG PCA- OCH KNN-MODELLEN
# ============================================================
# PCA och KNN sparas i samma sklearn-pipeline så att exakt samma
# PCA-representation används vid senare utvärdering.
def build_knn_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "pca",
                PCA(
                    n_components=KNN_PCA_VARIANCE,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=KNN_N_NEIGHBORS,
                    weights=KNN_WEIGHTS,
                    metric=KNN_METRIC,
                    algorithm=KNN_ALGORITHM,
                    n_jobs=KNN_N_JOBS,
                ),
            ),
        ]
    )
