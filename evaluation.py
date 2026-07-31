"""Evaluate a selected Zoodle model on the prepared test data."""

from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from cnn_model import predict_cnn_in_batches
from knn_model import prepare_knn_images
from settings import (
    ANIMAL_CLASSES,
    CNN_BATCH_SIZE,
    CNN_EVALUATION_OUTPUT_DIR,
    IMAGE_SIZE,
    KNN_DATA_FRACTION,
    KNN_EVALUATION_OUTPUT_DIR,
    KNN_MODEL_OUTPUT_DIR,
    KNN_PREDICTION_BATCH_SIZE,
    KNN_TRAINING_OUTPUT_DIR,
    MISCLASSIFIED_EXAMPLES,
    MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TOP_CONFUSIONS,
    TRAINING_OUTPUT_DIR,
)

CNN_MODEL_TYPE = "CNN"
KNN_MODEL_TYPE = "PCA + KNN"


# ============================================================
# 1. UTVÄRDERA MODELLEN
# ============================================================
# Användaren väljer modellkörning innan testdatan läses och utvärderas.
def evaluate_model() -> bool:
    selection = _select_checkpoint()
    if selection is None:
        return False

    model_type, model_path = selection
    checkpoint_id = model_path.parent.name
    evaluation_root = (
        CNN_EVALUATION_OUTPUT_DIR
        if model_type == CNN_MODEL_TYPE
        else KNN_EVALUATION_OUTPUT_DIR
    )
    output_dir = evaluation_root / checkpoint_id

    if output_dir.exists():
        print(
            f"\nCheckpoint {checkpoint_id} har redan utvärderats.\n"
            f"Befintligt resultat: {output_dir}\n"
            "Ta bort resultatmappen manuellt om modellen ska utvärderas på nytt."
        )
        return False

    test_data = _load_test_data()
    if test_data is None:
        return False

    test_images, test_labels = test_data
    del test_data
    sample_indices = np.arange(len(test_labels))

    if model_type == CNN_MODEL_TYPE:
        model = _load_model(model_path)
    else:
        model = _load_knn_model(model_path)

    if model is None:
        return False

    if model_type == KNN_MODEL_TYPE:
        selected_data = _select_knn_test_data(
            test_images,
            test_labels,
            sample_indices,
        )
        if selected_data is None:
            return False
        test_images, test_labels, sample_indices = selected_data

    print("\n========================================")
    print("MODELLUTVÄRDERING STARTAR")
    print("========================================")
    print(f"Modelltyp: {model_type}")
    print(f"Checkpoint: {checkpoint_id}")
    print(f"Testbilder: {len(test_images)}")

    # --------------------------------------------------------
    # 1.1 Skapa prediktioner
    # --------------------------------------------------------
    probabilities = _create_predictions(
        model=model,
        model_type=model_type,
        test_images=test_images,
    )
    if probabilities is None:
        return False

    expected_shape = (len(test_images), len(ANIMAL_CLASSES))
    if probabilities.shape != expected_shape or not np.all(np.isfinite(probabilities)):
        print(
            f"\nModellen gav ogiltig utdata: {probabilities.shape}. "
            f"Förväntad form är {expected_shape}."
        )
        return False

    predicted_labels = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted_labels == test_labels

    loss_values = tf.keras.losses.sparse_categorical_crossentropy(
        test_labels,
        probabilities,
    )
    test_loss = float(tf.reduce_mean(loss_values).numpy())
    test_accuracy = float(correct.mean())

    # --------------------------------------------------------
    # 1.2 Beräkna klassresultat och förväxlingar
    # --------------------------------------------------------
    class_metrics, report_table = _create_class_metrics(
        test_labels,
        predicted_labels,
    )
    roc_table, roc_curves = _create_roc_results(test_labels, probabilities)
    average_precision_table, precision_recall_curves = (
        _create_precision_recall_results(test_labels, probabilities)
    )
    confusion_counts = confusion_matrix(
        test_labels,
        predicted_labels,
        labels=np.arange(len(ANIMAL_CLASSES)),
    )
    top_confusions = _create_top_confusions(confusion_counts)

    predictions = pd.DataFrame(
        {
            "sample_index": sample_indices,
            "true_label": test_labels,
            "true_animal": np.array(ANIMAL_CLASSES)[test_labels],
            "predicted_label": predicted_labels,
            "predicted_animal": np.array(ANIMAL_CLASSES)[predicted_labels],
            "confidence": confidence,
            "correct": correct,
        }
    )

    # --------------------------------------------------------
    # 1.3 Skapa checkpointens resultatmapp
    # --------------------------------------------------------
    output_dir.mkdir(parents=True)

    roc_table.to_csv(output_dir / "roc_auc.csv", index=False)
    average_precision_table.to_csv(
        output_dir / "average_precision.csv",
        index=False,
    )
    top_confusions.to_csv(output_dir / "top_confusions.csv", index=False)
    predictions.to_csv(output_dir / "predictions.csv", index=False)

    confusion_table = pd.DataFrame(
        confusion_counts,
        index=ANIMAL_CLASSES,
        columns=ANIMAL_CLASSES,
    )
    confusion_table.to_csv(
        output_dir / "confusion_matrix.csv",
        index_label="true_animal",
    )

    # --------------------------------------------------------
    # 1.4 Spara tydliga figurer
    # --------------------------------------------------------
    _save_confusion_matrix(
        confusion_counts,
        output_dir / "confusion_matrix.png",
    )
    _save_class_performance(
        report_table,
        output_dir / "class_performance.png",
    )
    _save_roc_curve(
        roc_curves,
        output_dir / "roc_curve.png",
    )
    _save_roc_auc_per_class(
        roc_table,
        output_dir / "roc_auc_per_class.png",
    )
    _save_precision_recall_curve(
        precision_recall_curves,
        output_dir / "precision_recall_curve.png",
    )
    _save_average_precision_per_class(
        average_precision_table,
        precision_recall_curves,
        output_dir / "average_precision_per_class.png",
    )
    _save_top_confusions(
        top_confusions,
        output_dir / "top_confusions.png",
    )
    _save_misclassified_examples(
        images=test_images,
        true_labels=test_labels,
        predicted_labels=predicted_labels,
        confidence=confidence,
        correct=correct,
        output_path=output_dir / "misclassified_examples.png",
    )

    _save_summary(
        checkpoint_id=checkpoint_id,
        model_type=model_type,
        model_path=model_path,
        test_size=len(test_images),
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        correct_count=int(correct.sum()),
        class_metrics=class_metrics,
        report_table=report_table,
        roc_table=roc_table,
        roc_curves=roc_curves,
        average_precision_table=average_precision_table,
        precision_recall_curves=precision_recall_curves,
        output_path=output_dir / "summary.txt",
    )

    print("\n========================================")
    print("MODELLUTVÄRDERINGEN ÄR KLAR")
    print("========================================")
    print(f"Testförlust: {test_loss:.4f}")
    print(f"Testträffsäkerhet: {test_accuracy:.2%}")
    print(f"Makro-F1: {class_metrics['macro_f1']:.4f}")
    print(f"Makro ROC-AUC: {roc_curves['macro_auc']:.4f}")
    print(f"Mikro ROC-AUC: {roc_curves['micro_auc']:.4f}")
    print(
        "Makro Average Precision: "
        f"{precision_recall_curves['macro_average_precision']:.4f}"
    )
    print(
        "Mikro Average Precision: "
        f"{precision_recall_curves['micro_average_precision']:.4f}"
    )
    average_precision_class_rows = average_precision_table[
        average_precision_table["label"] != ""
    ]
    best_average_precision_class = average_precision_class_rows.loc[
        average_precision_class_rows["average_precision"].idxmax()
    ]
    weakest_average_precision_class = average_precision_class_rows.loc[
        average_precision_class_rows["average_precision"].idxmin()
    ]
    print(
        "Starkaste AP-klass: "
        f"{best_average_precision_class['animal']} "
        f"({best_average_precision_class['average_precision']:.4f})"
    )
    print(
        "Svagaste AP-klass: "
        f"{weakest_average_precision_class['animal']} "
        f"({weakest_average_precision_class['average_precision']:.4f})"
    )
    print(f"Resultat: {output_dir}")

    return True


# ============================================================
# 2. VÄLJ CHECKPOINT
# ============================================================
# CNN- och KNN-modeller sorteras tillsammans med den senaste körningen först.
def _select_checkpoint() -> tuple[str, Path] | None:
    checkpoints = [
        (CNN_MODEL_TYPE, path)
        for path in MODEL_OUTPUT_DIR.glob("*/best_model.keras")
    ]
    checkpoints.extend(
        (KNN_MODEL_TYPE, path)
        for path in KNN_MODEL_OUTPUT_DIR.glob("*/knn_model.joblib")
    )
    checkpoints = sorted(
        checkpoints,
        key=lambda item: item[1].parent.name,
        reverse=True,
    )

    if not checkpoints:
        print(
            "\nInga tränade modeller hittades i "
            f"{MODEL_OUTPUT_DIR} eller {KNN_MODEL_OUTPUT_DIR}."
        )
        print("Träna en modell innan utvärderingen startas.")
        return None

    print("\n========================================")
    print("VÄLJ MODELL FÖR UTVÄRDERING")
    print("========================================")

    for number, (model_type, checkpoint) in enumerate(checkpoints, start=1):
        checkpoint_id = checkpoint.parent.name
        details = _read_training_details(checkpoint_id, model_type)
        detail_text = f" | {details}" if details else ""
        print(f"{number}. [{model_type}] {checkpoint_id}{detail_text}")

    print("0. Avbryt")

    while True:
        choice = input("\nVälj modellkörning: ").strip()

        if choice == "0":
            print("\nUtvärderingen avbröts.")
            return None

        if choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
            selected_model = checkpoints[int(choice) - 1]
            model_type, selected_checkpoint = selected_model
            print(
                f"\nVald modell: {model_type} "
                f"({selected_checkpoint.parent.name})"
            )
            return selected_model

        print("\nOgiltigt val. Försök igen.")


# ============================================================
# 3. LÄS TRÄNINGSINFORMATION
# ============================================================
# En kort beskrivning hjälper användaren att skilja modellerna åt i menyn.
def _read_training_details(
    checkpoint_id: str,
    model_type: str = CNN_MODEL_TYPE,
) -> str:
    if model_type == CNN_MODEL_TYPE:
        summary_path = TRAINING_OUTPUT_DIR / checkpoint_id / "summary.txt"
        wanted_fields = {
            "Valideringsträffsäkerhet vid bästa epok": "validation_accuracy",
            "Dropout": "dropout",
            "Dataaugmentering": "augmentation",
            "Eager mode": "eager_mode",
        }
    else:
        summary_path = KNN_TRAINING_OUTPUT_DIR / checkpoint_id / "summary.txt"
        wanted_fields = {
            "Valideringsträffsäkerhet": "validation_accuracy",
            "Antal grannar": "neighbors",
            "Antal PCA-komponenter": "pca_components",
        }

    if not summary_path.exists():
        return ""

    try:
        summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    values = {}
    for line in summary_lines:
        if ":" not in line:
            continue

        field, value = line.split(":", maxsplit=1)
        if field in wanted_fields:
            values[wanted_fields[field]] = value.strip()

    details = []

    if "validation_accuracy" in values:
        try:
            accuracy = float(values["validation_accuracy"])
            details.append(f"validering {accuracy:.2%}")
        except ValueError:
            details.append(f"validering {values['validation_accuracy']}")

    if "dropout" in values:
        details.append(f"dropout {values['dropout']}")

    if "augmentation" in values:
        details.append(f"augmentering {values['augmentation']}")

    if "eager_mode" in values:
        details.append(f"eager mode {values['eager_mode']}")

    if "neighbors" in values:
        details.append(f"grannar {values['neighbors']}")

    if "pca_components" in values:
        details.append(f"PCA {values['pca_components']} komponenter")

    return " | ".join(details)


# ============================================================
# 4. LADDA TESTDATA
# ============================================================
# Testspliten kontrolleras innan modellen används och resultatmappar skapas.
def _load_test_data() -> tuple[np.ndarray, np.ndarray] | None:
    test_path = PROCESSED_DATA_DIR / "test.npz"

    if not test_path.exists():
        print(f"\nTestdatan saknas: {test_path}")
        print("Kör dataförberedelsen innan modellutvärderingen startas.")
        return None

    try:
        with np.load(test_path) as data:
            if "images" not in data or "labels" not in data:
                print("\nTestfilen måste innehålla images och labels.")
                return None

            images = data["images"]
            labels = data["labels"]
    except (OSError, ValueError, EOFError) as error:
        print(f"\nKunde inte läsa {test_path}: {error}")
        return None

    expected_image_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)

    if images.ndim != 4 or images.shape[1:] != expected_image_shape:
        print(
            f"\nFel bildform i testdatan: {images.shape}. "
            f"Förväntad form är (antal bilder, {IMAGE_SIZE}, {IMAGE_SIZE}, 1)."
        )
        return None

    if images.dtype != np.uint8:
        print(
            f"\nFel datatyp i testdatan: {images.dtype}. "
            "Förväntad datatyp är uint8."
        )
        return None

    if labels.ndim != 1 or len(labels) != len(images):
        print("\nEtiketterna i testdatan matchar inte antalet bilder.")
        return None

    if not np.issubdtype(labels.dtype, np.integer):
        print("\nEtiketterna i testdatan måste vara heltal.")
        return None

    if len(labels) == 0 or labels.min() < 0 or labels.max() >= len(ANIMAL_CLASSES):
        print("\nTestdatan innehåller ogiltiga etiketter.")
        return None

    class_counts = np.bincount(labels, minlength=len(ANIMAL_CLASSES))
    if np.any(class_counts == 0):
        missing_classes = [
            animal
            for label, animal in enumerate(ANIMAL_CLASSES)
            if class_counts[label] == 0
        ]
        print(f"\nTestdatan saknar följande klasser: {missing_classes}")
        return None

    return images, labels


# --------------------------------------------------------
# 4.1 Välj ett balanserat testurval för KNN
# --------------------------------------------------------
# Samma seed och andel används för varje klass så att urvalet kan återskapas.
def _select_knn_test_data(
    images: np.ndarray,
    labels: np.ndarray,
    sample_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not 0 < KNN_DATA_FRACTION <= 1:
        print("\nKNN_DATA_FRACTION måste vara större än 0 och högst 1.")
        return None

    class_counts = np.bincount(labels, minlength=len(ANIMAL_CLASSES))
    if len(np.unique(class_counts)) != 1:
        print("\nTestdatan måste ha lika många bilder i varje klass för KNN.")
        return None

    if KNN_DATA_FRACTION == 1.0:
        return images, labels, sample_indices

    samples_per_class = int(round(class_counts[0] * KNN_DATA_FRACTION))
    if samples_per_class < 1:
        print("\nKNN-andelen ger inga testbilder per klass.")
        return None

    random_generator = np.random.default_rng(RANDOM_STATE + 2)
    selected_indices = []

    for label in range(len(ANIMAL_CLASSES)):
        class_indices = np.flatnonzero(labels == label)
        selected_indices.extend(
            random_generator.choice(
                class_indices,
                size=samples_per_class,
                replace=False,
            )
        )

    selected_indices = np.sort(np.asarray(selected_indices, dtype=np.int64))

    return (
        images[selected_indices],
        labels[selected_indices],
        sample_indices[selected_indices],
    )


# ============================================================
# 5. LADDA MODELLEN
# ============================================================
# Optimizerstatus behövs inte när CNN-modellen endast ska skapa prediktioner.
def _load_model(model_path: Path) -> tf.keras.Model | None:
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (OSError, ValueError, tf.errors.OpError) as error:
        print(f"\nKunde inte läsa modellen {model_path}: {error}")
        return None

    expected_input_shape = (None, IMAGE_SIZE, IMAGE_SIZE, 1)
    expected_output_shape = (None, len(ANIMAL_CLASSES))

    if model.input_shape != expected_input_shape:
        print(
            f"\nModellen har fel indataform: {model.input_shape}. "
            f"Förväntad form är {expected_input_shape}."
        )
        return None

    if model.output_shape != expected_output_shape:
        print(
            f"\nModellen har fel utdataform: {model.output_shape}. "
            f"Förväntad form är {expected_output_shape}."
        )
        return None

    return model


# --------------------------------------------------------
# 5.1 Ladda PCA- och KNN-modellen
# --------------------------------------------------------
# Den sparade pipelinen måste innehålla både en tränad PCA och en tränad KNN.
def _load_knn_model(model_path: Path) -> Pipeline | None:
    try:
        model = joblib.load(model_path)
    except Exception as error:
        print(f"\nKunde inte läsa modellen {model_path}: {error}")
        return None

    if not isinstance(model, Pipeline):
        print("\nKNN-modellen måste vara en sklearn-pipeline.")
        return None

    pca_model = model.named_steps.get("pca")
    knn_model = model.named_steps.get("knn")

    if not isinstance(pca_model, PCA) or not isinstance(
        knn_model,
        KNeighborsClassifier,
    ):
        print("\nKNN-pipelinen måste innehålla stegen pca och knn.")
        return None

    expected_features = IMAGE_SIZE * IMAGE_SIZE
    if getattr(model, "n_features_in_", None) != expected_features:
        print(
            f"\nKNN-modellen har fel antal indatavärden: "
            f"{getattr(model, 'n_features_in_', None)}. "
            f"Förväntat antal är {expected_features}."
        )
        return None

    expected_classes = np.arange(len(ANIMAL_CLASSES))
    model_classes = getattr(knn_model, "classes_", None)
    if model_classes is None or not np.array_equal(
        model_classes,
        expected_classes,
    ):
        print("\nKNN-modellen måste innehålla samtliga etiketter 0–47.")
        return None

    return model


# --------------------------------------------------------
# 5.2 Skapa prediktioner med vald modell
# --------------------------------------------------------
# CNN anropas direkt i eager mode medan KNN arbetar med normaliserade platta bilder.
def _create_predictions(
    model: tf.keras.Model | Pipeline,
    model_type: str,
    test_images: np.ndarray,
) -> np.ndarray | None:
    if model_type == CNN_MODEL_TYPE:
        try:
            return predict_cnn_in_batches(
                model=model,
                images=test_images,
                batch_size=CNN_BATCH_SIZE,
                progress_label="CNN-prediktioner",
            )
        except (OSError, ValueError, tf.errors.OpError) as error:
            print(f"\nModellen kunde inte skapa prediktioner: {error}")
            return None

    if KNN_PREDICTION_BATCH_SIZE < 1:
        print("\nKNN_PREDICTION_BATCH_SIZE måste vara minst 1.")
        return None

    try:
        prepared_images = prepare_knn_images(test_images)
        probability_batches = []

        for start in range(
            0,
            len(prepared_images),
            KNN_PREDICTION_BATCH_SIZE,
        ):
            stop = min(
                start + KNN_PREDICTION_BATCH_SIZE,
                len(prepared_images),
            )
            probability_batches.append(
                model.predict_proba(prepared_images[start:stop])
            )
            _print_progress_bar(
                current=stop,
                total=len(prepared_images),
                label="KNN-prediktioner",
            )

        return np.vstack(probability_batches)
    except (MemoryError, OSError, ValueError) as error:
        print(f"\nModellen kunde inte skapa prediktioner: {error}")
        return None


# --------------------------------------------------------
# 5.3 Visa förloppet för KNN-prediktionerna
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
# 6. BERÄKNA KLASSMÅTT
# ============================================================
# Precision, recall och F1 behålls som utvärderingsmått utan classification report.
def _create_class_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    labels = np.arange(len(ANIMAL_CLASSES))
    precision, recall, f1_score, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=labels,
            average="weighted",
            zero_division=0,
        )
    )

    report_table = pd.DataFrame(
        {
            "label": labels,
            "animal": ANIMAL_CLASSES,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support.astype(int),
        }
    )
    class_metrics = {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }

    return class_metrics, report_table


# ============================================================
# 7. BERÄKNA ROC
# ============================================================
# Varje klass behandlas som positiv mot projektets övriga djurklasser.
def _create_roc_results(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    binary_labels = label_binarize(
        true_labels,
        classes=np.arange(len(ANIMAL_CLASSES)),
    )
    class_curves = []
    rows = []

    for label, animal in enumerate(ANIMAL_CLASSES):
        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_labels[:, label],
            probabilities[:, label],
        )
        class_auc = auc(false_positive_rate, true_positive_rate)
        class_curves.append((false_positive_rate, true_positive_rate))
        rows.append(
            {
                "label": label,
                "animal": animal,
                "roc_auc": class_auc,
            }
        )

    micro_fpr, micro_tpr, _ = roc_curve(
        binary_labels.ravel(),
        probabilities.ravel(),
    )
    micro_auc = auc(micro_fpr, micro_tpr)

    all_false_positive_rates = np.unique(
        np.concatenate([curve[0] for curve in class_curves])
    )
    mean_true_positive_rate = np.zeros_like(all_false_positive_rates)

    for false_positive_rate, true_positive_rate in class_curves:
        mean_true_positive_rate += np.interp(
            all_false_positive_rates,
            false_positive_rate,
            true_positive_rate,
        )

    mean_true_positive_rate /= len(ANIMAL_CLASSES)
    macro_auc = auc(all_false_positive_rates, mean_true_positive_rate)

    rows.extend(
        [
            {
                "label": "",
                "animal": "micro_average",
                "roc_auc": micro_auc,
            },
            {
                "label": "",
                "animal": "macro_average",
                "roc_auc": macro_auc,
            },
        ]
    )

    roc_curves = {
        "micro_fpr": micro_fpr,
        "micro_tpr": micro_tpr,
        "micro_auc": micro_auc,
        "macro_fpr": all_false_positive_rates,
        "macro_tpr": mean_true_positive_rate,
        "macro_auc": macro_auc,
    }

    return pd.DataFrame(rows), roc_curves


# ============================================================
# 8. BERÄKNA PRECISION-RECALL
# ============================================================
# Varje klass behandlas som positiv mot projektets övriga djurklasser.
def _create_precision_recall_results(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    binary_labels = label_binarize(
        true_labels,
        classes=np.arange(len(ANIMAL_CLASSES)),
    )
    recall_grid = np.linspace(0.0, 1.0, 1001)
    mean_precision = np.zeros_like(recall_grid)
    rows = []

    for label, animal in enumerate(ANIMAL_CLASSES):
        precision, recall, _ = precision_recall_curve(
            binary_labels[:, label],
            probabilities[:, label],
        )
        class_average_precision = average_precision_score(
            binary_labels[:, label],
            probabilities[:, label],
        )

        # Recall returneras i fallande ordning och vänds före interpoleringen.
        # Makrokurvan är ett visuellt medelvärde på en gemensam recall-skala.
        mean_precision += np.interp(
            recall_grid,
            recall[::-1],
            precision[::-1],
        )
        rows.append(
            {
                "label": label,
                "animal": animal,
                "average_precision": class_average_precision,
            }
        )

    mean_precision /= len(ANIMAL_CLASSES)
    micro_precision, micro_recall, _ = precision_recall_curve(
        binary_labels.ravel(),
        probabilities.ravel(),
    )
    micro_average_precision = average_precision_score(
        binary_labels,
        probabilities,
        average="micro",
    )
    # Macro Average Precision är sklearn-måttet och beräknas separat från
    # den visuellt sammanvägda makrokurvan ovan.
    macro_average_precision = average_precision_score(
        binary_labels,
        probabilities,
        average="macro",
    )

    rows.extend(
        [
            {
                "label": "",
                "animal": "micro_average",
                "average_precision": micro_average_precision,
            },
            {
                "label": "",
                "animal": "macro_average",
                "average_precision": macro_average_precision,
            },
        ]
    )

    precision_recall_curves = {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_average_precision": micro_average_precision,
        "macro_precision": mean_precision,
        "macro_recall": recall_grid,
        "macro_average_precision": macro_average_precision,
        "baseline": float(binary_labels.mean()),
    }

    return pd.DataFrame(rows), precision_recall_curves


# ============================================================
# 9. HITTA VANLIGA FÖRVÄXLINGAR
# ============================================================
# Diagonalen tas bort så att tabellen endast visar felaktiga prediktioner.
def _create_top_confusions(confusion_counts: np.ndarray) -> pd.DataFrame:
    error_counts = confusion_counts.copy()
    np.fill_diagonal(error_counts, 0)
    sorted_positions = np.argsort(error_counts, axis=None)[::-1]
    rows = []

    for position in sorted_positions:
        true_label, predicted_label = np.unravel_index(
            position,
            error_counts.shape,
        )
        count = int(error_counts[true_label, predicted_label])

        if count == 0 or len(rows) == TOP_CONFUSIONS:
            break

        class_total = int(confusion_counts[true_label].sum())
        rows.append(
            {
                "true_label": true_label,
                "true_animal": ANIMAL_CLASSES[true_label],
                "predicted_label": predicted_label,
                "predicted_animal": ANIMAL_CLASSES[predicted_label],
                "count": count,
                "percentage_of_true_class": count / class_total,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "true_label",
            "true_animal",
            "predicted_label",
            "predicted_animal",
            "count",
            "percentage_of_true_class",
        ],
    )


# ============================================================
# 10. SPARA CONFUSION MATRIX
# ============================================================
# Matrisen normaliseras per verklig klass och visas utan text i varje ruta.
def _save_confusion_matrix(
    confusion_counts: np.ndarray,
    output_path: Path,
) -> None:
    class_totals = confusion_counts.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        confusion_counts,
        class_totals,
        where=class_totals != 0,
    )

    fig, ax = plt.subplots(figsize=(18, 16))
    image = ax.imshow(
        normalized_matrix,
        cmap="Blues",
        vmin=0,
        vmax=max(float(normalized_matrix.max()), 0.01),
    )
    ax.set_title("Normaliserad förväxlingsmatris för testdatan", fontsize=16)
    ax.set_xlabel("Predikterad djurklass")
    ax.set_ylabel("Verklig djurklass")
    ax.set_xticks(np.arange(len(ANIMAL_CLASSES)), ANIMAL_CLASSES)
    ax.set_yticks(np.arange(len(ANIMAL_CLASSES)), ANIMAL_CLASSES)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Andel av verklig klass")
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 11. SPARA KLASSRESULTAT
# ============================================================
# Klasserna sorteras efter F1 så att styrkor och svagheter syns tydligt.
def _save_class_performance(
    report_table: pd.DataFrame,
    output_path: Path,
) -> None:
    class_rows = report_table[report_table["label"] != ""].copy()
    class_rows = class_rows.sort_values("f1_score")

    fig, ax = plt.subplots(figsize=(11, 14))
    ax.barh(class_rows["animal"], class_rows["f1_score"], color="#4C78A8")
    ax.set_title("F1-resultat per djurklass")
    ax.set_xlabel("F1")
    ax.set_ylabel("Djurklass")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 12. SPARA ROC-KURVOR
# ============================================================
# Mikro- och makrogenomsnitt visas utan separata kurvor för alla 48 klasser.
def _save_roc_curve(roc_curves: dict, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(
        roc_curves["micro_fpr"],
        roc_curves["micro_tpr"],
        color="#4C78A8",
        linewidth=2,
        label=f"Mikrogenomsnitt (AUC {roc_curves['micro_auc']:.4f})",
    )
    ax.plot(
        roc_curves["macro_fpr"],
        roc_curves["macro_tpr"],
        color="#F58518",
        linewidth=2,
        label=f"Makrogenomsnitt (AUC {roc_curves['macro_auc']:.4f})",
    )
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", label="Slumpnivå")
    ax.set_title("ROC-kurvor för testdatan")
    ax.set_xlabel("Andel falskt positiva")
    ax.set_ylabel("Andel sant positiva")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 13. SPARA ROC-AUC PER KLASS
# ============================================================
# Det sorterade diagrammet visar tydligt vilka djurklasser som är svagast.
def _save_roc_auc_per_class(
    roc_table: pd.DataFrame,
    output_path: Path,
) -> None:
    class_rows = roc_table[roc_table["label"] != ""].copy()
    class_rows = class_rows.sort_values("roc_auc")

    fig, ax = plt.subplots(figsize=(11, 14))
    ax.barh(class_rows["animal"], class_rows["roc_auc"], color="#54A24B")
    ax.axvline(0.5, color="#777777", linestyle="--", label="Slumpnivå")
    ax.set_title("ROC-AUC per djurklass")
    ax.set_xlabel("ROC-AUC")
    ax.set_ylabel("Djurklass")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 14. SPARA PRECISION-RECALL-KURVOR
# ============================================================
# Mikro- och makrogenomsnitt visas utan separata kurvor för alla 48 klasser.
def _save_precision_recall_curve(
    precision_recall_curves: dict,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.step(
        precision_recall_curves["micro_recall"],
        precision_recall_curves["micro_precision"],
        color="#4C78A8",
        linewidth=2,
        where="post",
        label=(
            "Mikrogenomsnitt "
            f"(AP {precision_recall_curves['micro_average_precision']:.4f})"
        ),
    )
    ax.plot(
        precision_recall_curves["macro_recall"],
        precision_recall_curves["macro_precision"],
        color="#F58518",
        linewidth=2,
        label=(
            "Visuellt makrogenomsnitt "
            f"(AP {precision_recall_curves['macro_average_precision']:.4f})"
        ),
    )
    ax.axhline(
        precision_recall_curves["baseline"],
        color="#777777",
        linestyle="--",
        label=f"Baslinje ({precision_recall_curves['baseline']:.2%})",
    )
    ax.set_title("Precision–Recall-kurvor för testdatan")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 15. SPARA AVERAGE PRECISION PER KLASS
# ============================================================
# Alla klasser visas i ett sorterat diagram i stället för 48 separata kurvor.
def _save_average_precision_per_class(
    average_precision_table: pd.DataFrame,
    precision_recall_curves: dict,
    output_path: Path,
) -> None:
    class_rows = average_precision_table[
        average_precision_table["label"] != ""
    ].copy()
    class_rows = class_rows.sort_values("average_precision")
    macro_average_precision = precision_recall_curves["macro_average_precision"]

    fig, ax = plt.subplots(figsize=(11, 14))
    ax.barh(
        class_rows["animal"],
        class_rows["average_precision"],
        color="#B279A2",
    )
    ax.axvline(
        macro_average_precision,
        color="#777777",
        linestyle="--",
        label=f"Makrogenomsnitt ({macro_average_precision:.4f})",
    )
    ax.set_title("Average Precision per djurklass")
    ax.set_xlabel("Average Precision")
    ax.set_ylabel("Djurklass")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 16. SPARA VANLIGA FÖRVÄXLINGAR
# ============================================================
# Diagrammet visar riktningen från verklig klass till modellens gissning.
def _save_top_confusions(
    top_confusions: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))

    if top_confusions.empty:
        ax.text(0.5, 0.5, "Inga felklassificeringar hittades.", ha="center")
        ax.axis("off")
    else:
        plot_data = top_confusions.iloc[::-1].copy()
        labels = (
            plot_data["true_animal"] + " → " + plot_data["predicted_animal"]
        )
        percentages = plot_data["percentage_of_true_class"] * 100
        ax.barh(labels, percentages, color="#E45756")
        ax.set_title("Vanligaste förväxlingarna")
        ax.set_xlabel("Andel av den verkliga klassen (%)")
        ax.set_ylabel("Verklig klass → predikterad klass")
        ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 17. SPARA FELKLASSIFICERADE EXEMPEL
# ============================================================
# De säkraste felaktiga gissningarna visar var modellen är mest övertygad men har fel.
def _save_misclassified_examples(
    images: np.ndarray,
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    confidence: np.ndarray,
    correct: np.ndarray,
    output_path: Path,
) -> None:
    incorrect_indexes = np.flatnonzero(~correct)
    sorted_indexes = incorrect_indexes[
        np.argsort(confidence[incorrect_indexes])[::-1]
    ]
    selected_indexes = sorted_indexes[:MISCLASSIFIED_EXAMPLES]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for ax in axes.flat:
        ax.axis("off")

    if len(selected_indexes) == 0:
        axes.flat[0].text(
            0.5,
            0.5,
            "Inga felklassificerade bilder hittades.",
            ha="center",
            va="center",
        )
    else:
        for ax, image_index in zip(axes.flat, selected_indexes):
            true_animal = ANIMAL_CLASSES[int(true_labels[image_index])]
            predicted_animal = ANIMAL_CLASSES[int(predicted_labels[image_index])]
            ax.imshow(images[image_index].squeeze(), cmap="gray", vmin=0, vmax=255)
            ax.set_title(
                f"Rätt: {true_animal}\n"
                f"Gissning: {predicted_animal} "
                f"({confidence[image_index]:.0%})",
                fontsize=9,
            )
            ax.axis("off")

    fig.suptitle("Säkra men felaktiga prediktioner", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ============================================================
# 18. SPARA SAMMANFATTNING
# ============================================================
# Sammanfattningen samlar de viktigaste testresultaten i ett läsbart format.
def _save_summary(
    checkpoint_id: str,
    model_path: Path,
    test_size: int,
    test_loss: float,
    test_accuracy: float,
    correct_count: int,
    class_metrics: dict,
    report_table: pd.DataFrame,
    roc_table: pd.DataFrame,
    roc_curves: dict,
    average_precision_table: pd.DataFrame,
    precision_recall_curves: dict,
    output_path: Path,
    model_type: str = CNN_MODEL_TYPE,
) -> None:
    class_rows = report_table[report_table["label"] != ""]
    best_class = class_rows.loc[class_rows["f1_score"].idxmax()]
    weakest_class = class_rows.loc[class_rows["f1_score"].idxmin()]
    roc_class_rows = roc_table[roc_table["label"] != ""]
    best_roc_class = roc_class_rows.loc[roc_class_rows["roc_auc"].idxmax()]
    weakest_roc_class = roc_class_rows.loc[roc_class_rows["roc_auc"].idxmin()]
    average_precision_class_rows = average_precision_table[
        average_precision_table["label"] != ""
    ]
    best_average_precision_class = average_precision_class_rows.loc[
        average_precision_class_rows["average_precision"].idxmax()
    ]
    weakest_average_precision_class = average_precision_class_rows.loc[
        average_precision_class_rows["average_precision"].idxmin()
    ]

    summary_lines = [
        "MODELLUTVÄRDERING - ZOODLE",
        "========================================",
        f"Modelltyp: {model_type}",
        f"Testdataandel: "
        f"{KNN_DATA_FRACTION if model_type == KNN_MODEL_TYPE else 1.0:.2%}",
        f"Checkpoint: {checkpoint_id}",
        f"Modell: {model_path}",
        f"Testbilder: {test_size}",
        f"Antal klasser: {len(ANIMAL_CLASSES)}",
        f"Rätt klassificerade: {correct_count}",
        f"Felklassificerade: {test_size - correct_count}",
        f"Testförlust: {test_loss:.4f}",
        f"Testträffsäkerhet: {test_accuracy:.4f}",
        f"Makroprecision: {class_metrics['macro_precision']:.4f}",
        f"Makro-recall: {class_metrics['macro_recall']:.4f}",
        f"Makro-F1: {class_metrics['macro_f1']:.4f}",
        f"Viktad F1: {class_metrics['weighted_f1']:.4f}",
        f"Makro ROC-AUC: {roc_curves['macro_auc']:.4f}",
        f"Mikro ROC-AUC: {roc_curves['micro_auc']:.4f}",
        "Makro Average Precision: "
        f"{precision_recall_curves['macro_average_precision']:.4f}",
        "Mikro Average Precision: "
        f"{precision_recall_curves['micro_average_precision']:.4f}",
        f"Starkaste klass: {best_class['animal']} "
        f"(F1 {best_class['f1_score']:.4f})",
        f"Svagaste klass: {weakest_class['animal']} "
        f"(F1 {weakest_class['f1_score']:.4f})",
        f"Starkaste ROC-AUC-klass: {best_roc_class['animal']} "
        f"({best_roc_class['roc_auc']:.4f})",
        f"Svagaste ROC-AUC-klass: {weakest_roc_class['animal']} "
        f"({weakest_roc_class['roc_auc']:.4f})",
        "Starkaste AP-klass: "
        f"{best_average_precision_class['animal']} "
        f"({best_average_precision_class['average_precision']:.4f})",
        "Svagaste AP-klass: "
        f"{weakest_average_precision_class['animal']} "
        f"({weakest_average_precision_class['average_precision']:.4f})",
    ]

    if model_type == CNN_MODEL_TYPE:
        summary_lines.insert(3, "Eager mode vid inference: Ja")

    output_path.write_text("\n".join(summary_lines), encoding="utf-8")


# ============================================================
# 19. STARTA MODELLUTVÄRDERINGEN
# ============================================================
# Funktionen kan köras direkt eller genom projektets pipeline-meny.
if __name__ == "__main__":
    evaluate_model()
