"""Train the Zoodle PCA and KNN model and save each run separately."""

from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report as sklearn_classification_report,
    f1_score,
)

from knn_model import build_knn_model, prepare_knn_images
from settings import (
    ANIMAL_CLASSES,
    IMAGE_SIZE,
    KNN_ALGORITHM,
    KNN_DATA_FRACTION,
    KNN_METRIC,
    KNN_MODEL_OUTPUT_DIR,
    KNN_N_JOBS,
    KNN_N_NEIGHBORS,
    KNN_PCA_VARIANCE,
    KNN_PREDICTION_BATCH_SIZE,
    KNN_TRAINING_OUTPUT_DIR,
    KNN_WEIGHTS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)


# ============================================================
# 1. TRÄNA PCA- OCH KNN-MODELLEN
# ============================================================
# Tränings- och valideringsdatan begränsas klassvis med samma seed.
# Testdatan används först när den sparade modellen utvärderas.
def train_knn_model() -> bool:
    if not 0 < KNN_DATA_FRACTION <= 1:
        print("\nKNN_DATA_FRACTION måste vara större än 0 och högst 1.")
        return False

    if KNN_PREDICTION_BATCH_SIZE <= 0:
        print("\nKNN_PREDICTION_BATCH_SIZE måste vara större än 0.")
        return False

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
    del train_data, validation_data

    # --------------------------------------------------------
    # 1.1 Välj ett balanserat urval
    # --------------------------------------------------------
    try:
        train_images, train_labels = _select_balanced_sample(
            train_images,
            train_labels,
            KNN_DATA_FRACTION,
        )
        validation_images, validation_labels = _select_balanced_sample(
            validation_images,
            validation_labels,
            KNN_DATA_FRACTION,
        )
    except ValueError as error:
        print(f"\nKunde inte skapa KNN-urvalet: {error}")
        return False

    train_images = prepare_knn_images(train_images)
    validation_images = prepare_knn_images(validation_images)

    run_id = _create_run_id()
    model_path = KNN_MODEL_OUTPUT_DIR / run_id / "knn_model.joblib"
    output_run_dir = KNN_TRAINING_OUTPUT_DIR / run_id

    print("\n========================================")
    print("TRÄNING AV PCA + KNN STARTAR")
    print("========================================")
    print(f"Körnings-ID: {run_id}")
    print(f"Dataandel: {KNN_DATA_FRACTION:.0%}")
    print(f"Träningsbilder: {len(train_images)}")
    print(f"Valideringsbilder: {len(validation_images)}")
    print(f"PCA-varians: {KNN_PCA_VARIANCE:.0%}")
    print(f"Antal grannar: {KNN_N_NEIGHBORS}")

    # --------------------------------------------------------
    # 1.2 Träna modellen
    # --------------------------------------------------------
    model = build_knn_model()
    training_start = perf_counter()

    try:
        model.fit(train_images, train_labels)
    except (MemoryError, TypeError, ValueError) as error:
        print(f"\nPCA- och KNN-modellen kunde inte tränas: {error}")
        return False

    training_time = perf_counter() - training_start

    # --------------------------------------------------------
    # 1.3 Skapa prediktioner för valideringsdatan
    # --------------------------------------------------------
    prediction_start = perf_counter()

    try:
        validation_predictions = _predict_in_batches(model, validation_images)
    except (MemoryError, TypeError, ValueError) as error:
        print(f"\nValideringsdatan kunde inte klassificeras: {error}")
        return False

    prediction_time = perf_counter() - prediction_start
    validation_accuracy = accuracy_score(
        validation_labels,
        validation_predictions,
    )
    validation_macro_f1 = f1_score(
        validation_labels,
        validation_predictions,
        labels=np.arange(len(ANIMAL_CLASSES)),
        average="macro",
        zero_division=0,
    )

    report_text, report_table = _create_classification_report(
        validation_labels,
        validation_predictions,
    )

    # --------------------------------------------------------
    # 1.4 Spara modellen och resultatet
    # --------------------------------------------------------
    try:
        model_path.parent.mkdir(parents=True)
        output_run_dir.mkdir(parents=True)
        joblib.dump(model, model_path)
        (output_run_dir / "classification_report.txt").write_text(
            report_text,
            encoding="utf-8",
        )
        report_table.to_csv(
            output_run_dir / "classification_report.csv",
            index=False,
        )
        _save_summary(
            model=model,
            run_id=run_id,
            train_size=len(train_images),
            validation_size=len(validation_images),
            validation_accuracy=validation_accuracy,
            validation_macro_f1=validation_macro_f1,
            training_time=training_time,
            prediction_time=prediction_time,
            model_path=model_path,
            output_path=output_run_dir / "summary.txt",
        )
    except (OSError, ValueError) as error:
        print(f"\nModellen eller träningsresultatet kunde inte sparas: {error}")
        return False

    print(f"\n{report_text}")
    print("\n========================================")
    print("TRÄNINGEN AV PCA + KNN ÄR KLAR")
    print("========================================")
    print(f"Valideringsträffsäkerhet: {validation_accuracy:.2%}")
    print(f"Macro F1 på valideringsdata: {validation_macro_f1:.4f}")
    print(f"Antal PCA-komponenter: {model.named_steps['pca'].n_components_}")
    print(f"Modell: {model_path}")
    print(f"Träningsresultat: {output_run_dir}")

    return True


# ============================================================
# 2. LADDA DATASET
# ============================================================
# Varje split kontrolleras innan urvalet och innan resultatmappar skapas.
def _load_dataset(
    file_path: Path,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not file_path.exists():
        print(f"\n{dataset_name} saknas: {file_path}")
        print("Kör dataförberedelsen innan KNN-träningen startas.")
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
# 3. VÄLJ ETT BALANSERAT URVAL
# ============================================================
# Samma andel väljs från varje klass utan återläggning.
def _select_balanced_sample(
    images: np.ndarray,
    labels: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if fraction == 1.0:
        return images, labels

    random_generator = np.random.default_rng(RANDOM_STATE)
    selected_indexes = []

    for label in range(len(ANIMAL_CLASSES)):
        class_indexes = np.flatnonzero(labels == label)
        sample_size = int(round(len(class_indexes) * fraction))

        if sample_size < 1:
            raise ValueError(
                f"Andelen ger inga bilder för klassen {ANIMAL_CLASSES[label]}."
            )

        class_selection = random_generator.choice(
            class_indexes,
            size=sample_size,
            replace=False,
        )
        selected_indexes.append(class_selection)

    selected_indexes = np.concatenate(selected_indexes)
    selected_indexes = random_generator.permutation(selected_indexes)

    return images[selected_indexes], labels[selected_indexes]


# ============================================================
# 4. SKAPA PREDIKTIONER I BATCHER
# ============================================================
# Mindre batcher begränsar minnesanvändningen när KNN beräknar avstånd.
def _predict_in_batches(model, images: np.ndarray) -> np.ndarray:
    predictions = []
    total_images = len(images)

    for start in range(0, total_images, KNN_PREDICTION_BATCH_SIZE):
        stop = min(start + KNN_PREDICTION_BATCH_SIZE, total_images)
        predictions.append(model.predict(images[start:stop]))
        _print_progress_bar(
            current=stop,
            total=total_images,
            label="Valideringsprediktioner",
        )

    return np.concatenate(predictions).astype(int)


# --------------------------------------------------------
# 4.1 Visa förloppet för KNN-prediktionerna
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


# ============================================================
# 5. SKAPA CLASSIFICATION REPORT
# ============================================================
# Rapporten visar precision, recall och F1 för varje klass i valideringsdatan.
def _create_classification_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[str, pd.DataFrame]:
    report_text = sklearn_classification_report(
        true_labels,
        predicted_labels,
        labels=np.arange(len(ANIMAL_CLASSES)),
        target_names=ANIMAL_CLASSES,
        zero_division=0,
    )
    report_text = f"Classification report\n\n{report_text}"

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
# 6. SKAPA KÖRNINGS-ID
# ============================================================
# Tidsstämpeln gör att tidigare modeller och resultat behålls.
def _create_run_id() -> str:
    base_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = base_run_id
    number = 2

    while (KNN_MODEL_OUTPUT_DIR / run_id).exists() or (
        KNN_TRAINING_OUTPUT_DIR / run_id
    ).exists():
        run_id = f"{base_run_id}_{number}"
        number += 1

    return run_id


# ============================================================
# 7. SPARA SAMMANFATTNING
# ============================================================
# Sammanfattningen dokumenterar inställningarna och resultatet för varje körning.
def _save_summary(
    model,
    run_id: str,
    train_size: int,
    validation_size: int,
    validation_accuracy: float,
    validation_macro_f1: float,
    training_time: float,
    prediction_time: float,
    model_path: Path,
    output_path: Path,
) -> None:
    pca = model.named_steps["pca"]

    summary_lines = [
        "MODELLTRÄNING PCA + KNN - ZOODLE",
        "========================================",
        f"Körnings-ID: {run_id}",
        "Modelltyp: PCA + KNN",
        f"Seed för urval: {RANDOM_STATE}",
        f"Dataandel: {KNN_DATA_FRACTION:.2%}",
        f"Träningsbilder: {train_size}",
        f"Träningsbilder per klass: {train_size // len(ANIMAL_CLASSES)}",
        f"Valideringsbilder: {validation_size}",
        f"Valideringsbilder per klass: "
        f"{validation_size // len(ANIMAL_CLASSES)}",
        f"Antal klasser: {len(ANIMAL_CLASSES)}",
        f"PCA-mål för varians: {KNN_PCA_VARIANCE:.4f}",
        f"PCA bevarad varians: {pca.explained_variance_ratio_.sum():.4f}",
        f"Antal PCA-komponenter: {pca.n_components_}",
        f"Antal grannar: {KNN_N_NEIGHBORS}",
        f"Viktning: {KNN_WEIGHTS}",
        f"Avståndsmått: {KNN_METRIC}",
        f"Algoritm: {KNN_ALGORITHM}",
        f"Parallella jobb: {KNN_N_JOBS}",
        f"Prediktionsbatch: {KNN_PREDICTION_BATCH_SIZE}",
        f"Valideringsträffsäkerhet: {validation_accuracy:.4f}",
        f"Macro F1 på valideringsdata: {validation_macro_f1:.4f}",
        f"Träningstid (sekunder): {training_time:.2f}",
        f"Prediktionstid (sekunder): {prediction_time:.2f}",
        f"Sparad modell: {model_path}",
    ]

    output_path.write_text("\n".join(summary_lines), encoding="utf-8")


# ============================================================
# 8. STARTA KNN-TRÄNINGEN
# ============================================================
# Funktionen kan köras direkt eller genom projektets pipeline-meny.
if __name__ == "__main__":
    train_knn_model()
