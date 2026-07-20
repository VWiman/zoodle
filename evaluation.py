"""Evaluate a selected Zoodle checkpoint on the prepared test data."""

from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from settings import (
    ANIMAL_CLASSES,
    BATCH_SIZE,
    EVALUATION_OUTPUT_DIR,
    IMAGE_SIZE,
    MISCLASSIFIED_EXAMPLES,
    MODEL_OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    TOP_CONFUSIONS,
    TRAINING_OUTPUT_DIR,
)


# ============================================================
# 1. UTVÄRDERA MODELLEN
# ============================================================
# Användaren väljer checkpoint innan testdatan läses och utvärderas.
def evaluate_model() -> bool:
    model_path = _select_checkpoint()
    if model_path is None:
        return False

    test_data = _load_test_data()
    if test_data is None:
        return False

    model = _load_model(model_path)
    if model is None:
        return False

    test_images, test_labels = test_data
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    checkpoint_id = model_path.parent.name

    print("\n========================================")
    print("MODELLUTVÄRDERING STARTAR")
    print("========================================")
    print(f"Checkpoint: {checkpoint_id}")
    print(f"Testbilder: {len(test_images)}")

    # --------------------------------------------------------
    # 1.1 Skapa prediktioner
    # --------------------------------------------------------
    try:
        probabilities = model.predict(test_dataset, verbose=1)
    except (OSError, ValueError, tf.errors.OpError) as error:
        print(f"\nModellen kunde inte skapa prediktioner: {error}")
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
    report, report_table = _create_classification_report(
        test_labels,
        predicted_labels,
    )
    confusion_counts = confusion_matrix(
        test_labels,
        predicted_labels,
        labels=np.arange(len(ANIMAL_CLASSES)),
    )
    top_confusions = _create_top_confusions(confusion_counts)

    predictions = pd.DataFrame(
        {
            "true_label": test_labels,
            "true_animal": np.array(ANIMAL_CLASSES)[test_labels],
            "predicted_label": predicted_labels,
            "predicted_animal": np.array(ANIMAL_CLASSES)[predicted_labels],
            "confidence": confidence,
            "correct": correct,
        }
    )

    # --------------------------------------------------------
    # 1.3 Skapa en unik resultatmapp
    # --------------------------------------------------------
    evaluation_id = _create_evaluation_id(checkpoint_id)
    output_dir = EVALUATION_OUTPUT_DIR / checkpoint_id / evaluation_id
    output_dir.mkdir(parents=True)

    report_table.to_csv(output_dir / "classification_report.csv", index=False)
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
        evaluation_id=evaluation_id,
        model_path=model_path,
        test_size=len(test_images),
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        correct_count=int(correct.sum()),
        report=report,
        report_table=report_table,
        output_path=output_dir / "summary.txt",
    )

    print("\n========================================")
    print("MODELLUTVÄRDERINGEN ÄR KLAR")
    print("========================================")
    print(f"Testförlust: {test_loss:.4f}")
    print(f"Testträffsäkerhet: {test_accuracy:.2%}")
    print(f"Makro-F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Resultat: {output_dir}")

    return True


# ============================================================
# 2. VÄLJ CHECKPOINT
# ============================================================
# Modellerna sorteras med den senaste träningskörningen först.
def _select_checkpoint() -> Path | None:
    checkpoints = sorted(
        MODEL_OUTPUT_DIR.glob("*/best_model.keras"),
        key=lambda path: path.parent.name,
        reverse=True,
    )

    if not checkpoints:
        print(f"\nInga tränade modeller hittades i {MODEL_OUTPUT_DIR}.")
        print("Träna en modell innan utvärderingen startas.")
        return None

    print("\n========================================")
    print("VÄLJ CHECKPOINT FÖR UTVÄRDERING")
    print("========================================")

    for number, checkpoint in enumerate(checkpoints, start=1):
        checkpoint_id = checkpoint.parent.name
        details = _read_training_details(checkpoint_id)
        detail_text = f" | {details}" if details else ""
        print(f"{number}. {checkpoint_id}{detail_text}")

    print("0. Avbryt")

    while True:
        choice = input("\nVälj checkpoint: ").strip()

        if choice == "0":
            print("\nUtvärderingen avbröts.")
            return None

        if choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
            selected_checkpoint = checkpoints[int(choice) - 1]
            print(f"\nVald checkpoint: {selected_checkpoint.parent.name}")
            return selected_checkpoint

        print("\nOgiltigt val. Försök igen.")


# ============================================================
# 3. LÄS TRÄNINGSINFORMATION
# ============================================================
# En kort beskrivning hjälper användaren att skilja modellerna åt i menyn.
def _read_training_details(checkpoint_id: str) -> str:
    summary_path = TRAINING_OUTPUT_DIR / checkpoint_id / "summary.txt"
    if not summary_path.exists():
        return ""

    try:
        summary_lines = summary_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""

    values = {}
    wanted_fields = {
        "Valideringsträffsäkerhet vid bästa epok": "validation_accuracy",
        "Dropout": "dropout",
        "Dataaugmentering": "augmentation",
    }

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


# ============================================================
# 5. LADDA MODELLEN
# ============================================================
# Optimizerstatus behövs inte när modellen endast ska skapa prediktioner.
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


# ============================================================
# 6. SKAPA KLASSRAPPORT
# ============================================================
# Rapporten innehåller varje klass samt sammanfattande macro- och weighted-mått.
def _create_classification_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    report = classification_report(
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

    return report, pd.DataFrame(rows)


# ============================================================
# 7. HITTA VANLIGA FÖRVÄXLINGAR
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
# 8. SKAPA UTVÄRDERINGS-ID
# ============================================================
# Varje körning sparas separat under checkpointens egen mapp.
def _create_evaluation_id(checkpoint_id: str) -> str:
    base_evaluation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_id = base_evaluation_id
    number = 2

    while (
        EVALUATION_OUTPUT_DIR / checkpoint_id / evaluation_id
    ).exists():
        evaluation_id = f"{base_evaluation_id}_{number}"
        number += 1

    return evaluation_id


# ============================================================
# 9. SPARA CONFUSION MATRIX
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
# 10. SPARA KLASSRESULTAT
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
# 11. SPARA VANLIGA FÖRVÄXLINGAR
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
# 12. SPARA FELKLASSIFICERADE EXEMPEL
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
# 13. SPARA SAMMANFATTNING
# ============================================================
# Sammanfattningen samlar de viktigaste testresultaten i ett läsbart format.
def _save_summary(
    checkpoint_id: str,
    evaluation_id: str,
    model_path: Path,
    test_size: int,
    test_loss: float,
    test_accuracy: float,
    correct_count: int,
    report: dict,
    report_table: pd.DataFrame,
    output_path: Path,
) -> None:
    class_rows = report_table[report_table["label"] != ""]
    best_class = class_rows.loc[class_rows["f1_score"].idxmax()]
    weakest_class = class_rows.loc[class_rows["f1_score"].idxmin()]
    macro_metrics = report["macro avg"]
    weighted_metrics = report["weighted avg"]

    summary_lines = [
        "MODELLUTVÄRDERING - ZOODLE",
        "========================================",
        f"Checkpoint: {checkpoint_id}",
        f"Utvärderings-ID: {evaluation_id}",
        f"Modell: {model_path}",
        f"Testbilder: {test_size}",
        f"Antal klasser: {len(ANIMAL_CLASSES)}",
        f"Rätt klassificerade: {correct_count}",
        f"Felklassificerade: {test_size - correct_count}",
        f"Testförlust: {test_loss:.4f}",
        f"Testträffsäkerhet: {test_accuracy:.4f}",
        f"Makroprecision: {macro_metrics['precision']:.4f}",
        f"Makro-recall: {macro_metrics['recall']:.4f}",
        f"Makro-F1: {macro_metrics['f1-score']:.4f}",
        f"Viktad F1: {weighted_metrics['f1-score']:.4f}",
        f"Starkaste klass: {best_class['animal']} "
        f"(F1 {best_class['f1_score']:.4f})",
        f"Svagaste klass: {weakest_class['animal']} "
        f"(F1 {weakest_class['f1_score']:.4f})",
    ]

    output_path.write_text("\n".join(summary_lines), encoding="utf-8")


# ============================================================
# 14. STARTA MODELLUTVÄRDERINGEN
# ============================================================
# Funktionen kan köras direkt eller genom projektets pipeline-meny.
if __name__ == "__main__":
    evaluate_model()
