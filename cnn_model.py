"""Build and compile the CNN model used by the Zoodle project."""

import numpy as np
import tensorflow as tf

from settings import (
    ANIMAL_CLASSES,
    CNN_AUGMENTATION_ROTATION,
    CNN_AUGMENTATION_TRANSLATION,
    CNN_AUGMENTATION_ZOOM,
    CNN_DROPOUT_RATE,
    CNN_LEARNING_RATE,
    CNN_RUN_EAGERLY,
    CNN_USE_DATA_AUGMENTATION,
    IMAGE_SIZE,
    RANDOM_STATE,
)


# ============================================================
# 1. BYGG CNN-MODELLEN
# ============================================================
# Augmenteringen kan slås av och på för att jämföra olika träningskörningar.
def build_cnn_model(
    use_data_augmentation: bool = CNN_USE_DATA_AUGMENTATION,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    layers = [
        tf.keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        tf.keras.layers.Rescaling(1.0 / 255),
    ]

    # --------------------------------------------------------
    # 1.1 Lägg till valbar dataaugmentering
    # --------------------------------------------------------
    if use_data_augmentation:
        layers.extend(
            [
                tf.keras.layers.RandomRotation(
                    factor=CNN_AUGMENTATION_ROTATION,
                    fill_mode="constant",
                    seed=RANDOM_STATE,
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor=CNN_AUGMENTATION_TRANSLATION,
                    width_factor=CNN_AUGMENTATION_TRANSLATION,
                    fill_mode="constant",
                    seed=RANDOM_STATE + 1,
                ),
                tf.keras.layers.RandomZoom(
                    height_factor=CNN_AUGMENTATION_ZOOM,
                    width_factor=CNN_AUGMENTATION_ZOOM,
                    fill_mode="constant",
                    seed=RANDOM_STATE + 2,
                ),
            ]
        )

    # --------------------------------------------------------
    # 1.2 Lägg till modellens lager
    # --------------------------------------------------------
    layers.extend(
        [
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
            tf.keras.layers.GlobalMaxPooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(CNN_DROPOUT_RATE),
            tf.keras.layers.Dense(len(ANIMAL_CLASSES), activation="softmax"),
        ]
    )

    model = tf.keras.Sequential(layers, name="zoodle_cnn")

    # --------------------------------------------------------
    # 1.3 Kompilera modellen
    # --------------------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=CNN_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        run_eagerly=CNN_RUN_EAGERLY,
    )

    return model


# ============================================================
# 2. SKAPA CNN-PREDIKTIONER I EAGER MODE
# ============================================================
# Modellen anropas direkt i mindre batcher. Då används inte den graph-baserade
# predict-funktionen som kan ge felaktiga resultat med TensorFlow Metal.
def predict_cnn_in_batches(
    model: tf.keras.Model,
    images: np.ndarray,
    batch_size: int,
    progress_label: str,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("Batchstorleken måste vara minst 1.")

    if len(images) == 0:
        raise ValueError("Minst en bild behövs för att skapa prediktioner.")

    probability_batches = []

    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        probabilities = model(images[start:stop], training=False)
        probability_batches.append(probabilities.numpy())
        _print_progress_bar(
            current=stop,
            total=len(images),
            label=progress_label,
        )

    return np.vstack(probability_batches)


# --------------------------------------------------------
# 2.1 Visa förloppet för CNN-prediktionerna
# --------------------------------------------------------
# Progressbaren uppdateras efter varje batch och använder samma terminalrad.
def _print_progress_bar(current: int, total: int, label: str) -> None:
    progress = current / total
    bar_width = 30
    filled_width = int(bar_width * progress)
    bar = "█" * filled_width + "-" * (bar_width - filled_width)

    print(
        f"\r{label}: [{bar}] {current}/{total} ({progress:.0%})",
        end="",
        flush=True,
    )

    if current == total:
        print()
