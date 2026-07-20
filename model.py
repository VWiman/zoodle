"""Build and compile the CNN model used by the Zoodle project."""

import tensorflow as tf

from settings import (
    ANIMAL_CLASSES,
    AUGMENTATION_ROTATION,
    AUGMENTATION_TRANSLATION,
    AUGMENTATION_ZOOM,
    DROPOUT_RATE,
    IMAGE_SIZE,
    LEARNING_RATE,
    RANDOM_STATE,
    USE_DATA_AUGMENTATION,
)


# ============================================================
# 1. BYGG CNN-MODELLEN
# ============================================================
# Augmenteringen kan slås av och på för att jämföra olika träningskörningar.
def build_cnn_model(
    use_data_augmentation: bool = USE_DATA_AUGMENTATION,
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
                    factor=AUGMENTATION_ROTATION,
                    fill_mode="constant",
                    seed=RANDOM_STATE,
                ),
                tf.keras.layers.RandomTranslation(
                    height_factor=AUGMENTATION_TRANSLATION,
                    width_factor=AUGMENTATION_TRANSLATION,
                    fill_mode="constant",
                    seed=RANDOM_STATE + 1,
                ),
                tf.keras.layers.RandomZoom(
                    height_factor=AUGMENTATION_ZOOM,
                    width_factor=AUGMENTATION_ZOOM,
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
            tf.keras.layers.GlobalMaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(DROPOUT_RATE),
            tf.keras.layers.Dense(len(ANIMAL_CLASSES), activation="softmax"),
        ]
    )

    model = tf.keras.Sequential(layers, name="zoodle_cnn")

    # --------------------------------------------------------
    # 1.3 Kompilera modellen
    # --------------------------------------------------------
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
