"""Train the Zoodle CNN model and save each training run separately."""

from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report as sklearn_classification_report

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from model import build_cnn_model
from settings import (
    ANIMAL_CLASSES,
    AUGMENTATION_ROTATION,
    AUGMENTATION_TRANSLATION,
    AUGMENTATION_ZOOM,
    BATCH_SIZE,
    DROPOUT_RATE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    IMAGE_SIZE,
    LEARNING_RATE,
    MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TRAINING_OUTPUT_DIR,
    USE_DATA_AUGMENTATION,
)


# ============================================================
# 1. TRÄNA MODELLEN
# ============================================================
# Tränings- och valideringsdata används här. Testdatan sparas till utvärderingen.
def train_model() -> bool:
    train_path = PROCESSED_DATA_DIR / "train.npz"
    validation_path = PROCESSED_DATA_DIR / "validation.npz"

    train_data = _load_dataset(train_path, "Träningsdatan")
    if train_data is None:
        return False

    validation_data = _load_dataset(validation_path, "Valideringsdatan")
    if validation_data is None:
        return False

    train_images, train_labels = train_data
    validation_images, validation_labels = validation_data

    # --------------------------------------------------------
    # 1.1 Skapa TensorFlow-dataset
    # --------------------------------------------------------
    tf.keras.utils.set_random_seed(RANDOM_STATE)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)
    )
    train_dataset = train_dataset.shuffle(
        buffer_size=len(train_images),
        seed=RANDOM_STATE,
        reshuffle_each_iteration=True,
    )
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (validation_images, validation_labels)
    )
    validation_dataset = validation_dataset.batch(BATCH_SIZE).prefetch(
        tf.data.AUTOTUNE
    )

    # --------------------------------------------------------
    # 1.2 Skapa mappar för den aktuella körningen
    # --------------------------------------------------------
    run_id = _create_run_id()
    model_run_dir = MODEL_OUTPUT_DIR / run_id
    output_run_dir = TRAINING_OUTPUT_DIR / run_id
    model_run_dir.mkdir(parents=True)
    output_run_dir.mkdir(parents=True)

    model_path = model_run_dir / "best_model.keras"
    model = build_cnn_model(USE_DATA_AUGMENTATION)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("\n========================================")
    print("MODELLTRÄNING STARTAR")
    print("========================================")
    print(f"Körnings-ID: {run_id}")
    print(f"Träningsbilder: {len(train_images)}")
    print(f"Valideringsbilder: {len(validation_images)}")
    print(f"Dataaugmentering: {'Ja' if USE_DATA_AUGMENTATION else 'Nej'}")
    print(f"Maximalt antal epoker: {EPOCHS}")

    # --------------------------------------------------------
    # 1.3 Träna och spara den bästa modellen
    # --------------------------------------------------------
    try:
        history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
        )
    except (OSError, ValueError, tf.errors.OpError) as error:
        print(f"\nModellträningen kunde inte genomföras: {error}")
        return False

    # --------------------------------------------------------
    # 1.4 Spara träningsresultatet
    # --------------------------------------------------------
    history_table = pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.history["loss"]) + 1),
            "loss": history.history["loss"],
            "accuracy": history.history["accuracy"],
            "val_loss": history.history["val_loss"],
            "val_accuracy": history.history["val_accuracy"],
        }
    )
    history_table.to_csv(output_run_dir / "history.csv", index=False)

    best_index = int(history_table["val_loss"].idxmin())
    _save_training_figure(history_table, output_run_dir / "training_history.png")

    # --------------------------------------------------------
    # 1.5 Skapa classification report för valideringsdatan
    # --------------------------------------------------------
    try:
        best_model = tf.keras.models.load_model(model_path, compile=False)
        validation_probabilities = best_model.predict(
            validation_images,
            batch_size=BATCH_SIZE,
            verbose=1,
        )
    except (OSError, ValueError, tf.errors.OpError) as error:
        print(f"\nClassification report kunde inte skapas: {error}")
        return False

    expected_shape = (len(validation_images), len(ANIMAL_CLASSES))
    if validation_probabilities.shape != expected_shape or not np.all(
        np.isfinite(validation_probabilities)
    ):
        print(
            f"\nModellen gav ogiltig utdata: {validation_probabilities.shape}. "
            f"Förväntad form är {expected_shape}."
        )
        return False

    validation_predictions = validation_probabilities.argmax(axis=1)
    report_text, report_table = _create_classification_report(
        validation_labels,
        validation_predictions,
    )
    (output_run_dir / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )
    report_table.to_csv(output_run_dir / "classification_report.csv", index=False)

    print(f"\n{report_text}")

    _save_summary(
        model=model,
        run_id=run_id,
        train_size=len(train_images),
        validation_size=len(validation_images),
        history_table=history_table,
        best_index=best_index,
        model_path=model_path,
        output_path=output_run_dir / "summary.txt",
    )

    print("\n========================================")
    print("MODELLTRÄNINGEN ÄR KLAR")
    print("========================================")
    print(f"Bästa epok: {best_index + 1}")
    print(f"Bästa valideringsförlust: {history_table.loc[best_index, 'val_loss']:.4f}")
    print(f"Modell: {model_path}")
    print(f"Träningsresultat: {output_run_dir}")

    return True


# ============================================================
# 2. LADDA DATASET
# ============================================================
# Varje split kontrolleras innan träningen och innan nya resultatmappar skapas.
def _load_dataset(
    file_path: Path,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not file_path.exists():
        print(f"\n{dataset_name} saknas: {file_path}")
        print("Kör dataförberedelsen innan modellträningen startas.")
        return None

    try:
        with np.load(file_path) as data:
            if "images" not in data or "labels" not in data:
                print(f"\n{dataset_name} måste innehålla images och labels.")
                return None

            images = data["images"]
            labels = data["labels"]
    except (OSError, ValueError, EOFError) as error:
        print(f"\nKunde inte läsa {file_path}: {error}")
        return None

    expected_image_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)

    if images.ndim != 4 or images.shape[1:] != expected_image_shape:
        print(
            f"\nFel bildform i {dataset_name.lower()}: {images.shape}. "
            f"Förväntad form är (antal bilder, {IMAGE_SIZE}, {IMAGE_SIZE}, 1)."
        )
        return None

    if images.dtype != np.uint8:
        print(
            f"\nFel datatyp i {dataset_name.lower()}: {images.dtype}. "
            "Förväntad datatyp är uint8."
        )
        return None

    if labels.ndim != 1 or len(labels) != len(images):
        print(f"\nEtiketterna i {dataset_name.lower()} matchar inte antalet bilder.")
        return None

    if not np.issubdtype(labels.dtype, np.integer):
        print(f"\nEtiketterna i {dataset_name.lower()} måste vara heltal.")
        return None

    if len(labels) == 0 or labels.min() < 0 or labels.max() >= len(ANIMAL_CLASSES):
        print(f"\n{dataset_name} innehåller ogiltiga etiketter.")
        return None

    class_counts = np.bincount(labels, minlength=len(ANIMAL_CLASSES))
    if np.any(class_counts == 0):
        missing_classes = [
            animal
            for label, animal in enumerate(ANIMAL_CLASSES)
            if class_counts[label] == 0
        ]
        print(f"\n{dataset_name} saknar följande klasser: {missing_classes}")
        return None

    return images, labels


# ============================================================
# 3. SKAPA KÖRNINGS-ID
# ============================================================
# Tidsstämpeln gör att tidigare modeller och träningsresultat behålls.
def _create_run_id() -> str:
    base_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = base_run_id
    number = 2

    while (MODEL_OUTPUT_DIR / run_id).exists() or (
        TRAINING_OUTPUT_DIR / run_id
    ).exists():
        run_id = f"{base_run_id}_{number}"
        number += 1

    return run_id


# ============================================================
# 4. SPARA TRÄNINGSHISTORIK
# ============================================================
# Tränings- och valideringskurvorna sparas tillsammans för enkel jämförelse.
def _save_training_figure(history_table: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history_table["epoch"], history_table["loss"], label="Träning")
    axes[0].plot(
        history_table["epoch"],
        history_table["val_loss"],
        label="Validering",
    )
    axes[0].set_title("Förlust per epok")
    axes[0].set_xlabel("Epok")
    axes[0].set_ylabel("Förlust")
    axes[0].legend()

    axes[1].plot(
        history_table["epoch"],
        history_table["accuracy"],
        label="Träning",
    )
    axes[1].plot(
        history_table["epoch"],
        history_table["val_accuracy"],
        label="Validering",
    )
    axes[1].set_title("Träffsäkerhet per epok")
    axes[1].set_xlabel("Epok")
    axes[1].set_ylabel("Träffsäkerhet")
    axes[1].legend()

    fig.suptitle("Träningshistorik för CNN-modellen")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 5. SKAPA CLASSIFICATION REPORT
# ============================================================
# Rapporten visar precision, recall och F1 för varje klass i valideringsdatan.
def create_classification_report(
    y_true,
    y_pred,
    class_names=ANIMAL_CLASSES,
) -> str:
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    report = sklearn_classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    )

    return f"Classification report\n\n{report}"


def _create_classification_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    report_text = create_classification_report(true_labels, predicted_labels)
    report = sklearn_classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(len(ANIMAL_CLASSES)),
        target_names=ANIMAL_CLASSES,
        output_dict=True,
        zero_division=0,
    )

    rows = []

    for label, animal in enumerate(ANIMAL_CLASSES):
        metrics = report[animal]
        rows.append(
            {
                "label": label,
                "animal": animal,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": int(metrics["support"]),
            }
        )

    for report_key, row_name in [
        ("macro avg", "macro_average"),
        ("weighted avg", "weighted_average"),
    ]:
        metrics = report[report_key]
        rows.append(
            {
                "label": "",
                "animal": row_name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": int(metrics["support"]),
            }
        )

    return report_text, pd.DataFrame(rows)


# ============================================================
# 6. SPARA SAMMANFATTNING
# ============================================================
# Sammanfattningen dokumenterar inställningarna och resultatet för varje körning.
def _save_summary(
    model: tf.keras.Model,
    run_id: str,
    train_size: int,
    validation_size: int,
    history_table: pd.DataFrame,
    best_index: int,
    model_path: Path,
    output_path: Path,
) -> None:
    model_summary = []
    model.summary(print_fn=model_summary.append)

    summary_lines = [
        "MODELLTRÄNING - ZOODLE",
        "========================================",
        f"Körnings-ID: {run_id}",
        f"Dataaugmentering: {'Ja' if USE_DATA_AUGMENTATION else 'Nej'}",
        f"Rotation vid augmentering: {AUGMENTATION_ROTATION}",
        f"Förflyttning vid augmentering: {AUGMENTATION_TRANSLATION}",
        f"Zoom vid augmentering: {AUGMENTATION_ZOOM}",
        f"Dropout: {DROPOUT_RATE}",
        f"Träningsbilder: {train_size}",
        f"Valideringsbilder: {validation_size}",
        f"Antal klasser: {len(ANIMAL_CLASSES)}",
        f"Batchstorlek: {BATCH_SIZE}",
        f"Maximalt antal epoker: {EPOCHS}",
        f"Genomförda epoker: {len(history_table)}",
        f"Tålamod för early stopping: {EARLY_STOPPING_PATIENCE}",
        f"Inlärningshastighet: {LEARNING_RATE}",
        f"Bästa epok: {best_index + 1}",
        f"Bästa valideringsförlust: {history_table.loc[best_index, 'val_loss']:.4f}",
        f"Valideringsträffsäkerhet vid bästa epok: "
        f"{history_table.loc[best_index, 'val_accuracy']:.4f}",
        f"Sparad modell: {model_path}",
        "",
        "MODELLARKITEKTUR",
        "========================================",
        *model_summary,
    ]

    output_path.write_text("\n".join(summary_lines), encoding="utf-8")


# ============================================================
# 7. STARTA MODELLTRÄNINGEN
# ============================================================
# Funktionen kan köras direkt eller genom projektets pipeline-meny.
if __name__ == "__main__":
    train_model()
