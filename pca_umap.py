"""Analyze the prepared training images with PCA and UMAP."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from settings import (
    ANIMAL_CLASSES,
    ANIMAL_GROUPS,
    IMAGE_SIZE,
    PCA_OUTPUT_DIR,
    PCA_SAMPLES_PER_CLASS,
    PCA_VARIANCE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_OUTPUT_DIR,
    UMAP_SAMPLES_PER_CLASS,
)


# ============================================================
# 1. ANALYSERA MED PCA OCH UMAP
# ============================================================
# Endast träningsdatan används och båda analyserna får balanserade klassurval.
def run_pca_umap() -> bool:
    train_path = PROCESSED_DATA_DIR / "train.npz"

    if not train_path.exists():
        print(f"\nTräningsdatan saknas: {train_path}")
        print("Kör dataförberedelsen innan PCA och UMAP startas.")
        return False

    # --------------------------------------------------------
    # 1.1 Ladda och kontrollera träningsdatan
    # --------------------------------------------------------
    try:
        with np.load(train_path) as data:
            if "images" not in data or "labels" not in data:
                print("\nTräningsfilen måste innehålla images och labels.")
                return False

            images = data["images"]
            labels = data["labels"]
    except (OSError, ValueError, EOFError) as error:
        print(f"\nKunde inte läsa {train_path}: {error}")
        return False

    expected_image_shape = (IMAGE_SIZE, IMAGE_SIZE, 1)

    if images.ndim != 4 or images.shape[1:] != expected_image_shape:
        print(
            f"\nFel bildform: {images.shape}. "
            f"Förväntad form är (antal bilder, {IMAGE_SIZE}, {IMAGE_SIZE}, 1)."
        )
        return False

    if images.dtype != np.uint8:
        print(f"\nFel datatyp för bilderna: {images.dtype}. Förväntad datatyp är uint8.")
        return False

    if labels.ndim != 1 or len(labels) != len(images):
        print("\nEtiketterna har inte samma antal rader som bilderna.")
        return False

    if not np.issubdtype(labels.dtype, np.integer):
        print("\nEtiketterna måste vara heltal.")
        return False

    if len(labels) == 0 or labels.min() < 0 or labels.max() >= len(ANIMAL_CLASSES):
        print("\nTräningsdatan innehåller ogiltiga etiketter.")
        return False

    class_counts = np.bincount(labels, minlength=len(ANIMAL_CLASSES))

    if np.any(class_counts < PCA_SAMPLES_PER_CLASS):
        print(
            f"\nVarje klass måste innehålla minst "
            f"{PCA_SAMPLES_PER_CLASS} bilder för PCA-urvalet."
        )
        return False

    animal_to_group = {
        animal: group
        for group, animals in ANIMAL_GROUPS.items()
        for animal in animals
    }

    if any(animal not in animal_to_group for animal in ANIMAL_CLASSES):
        print("\nEn eller flera djurklasser saknar djurgrupp i inställningarna.")
        return False

    # --------------------------------------------------------
    # 1.2 Välj och normalisera bilder för PCA
    # --------------------------------------------------------
    random_generator = np.random.default_rng(RANDOM_STATE)
    pca_indexes = []

    for label in range(len(ANIMAL_CLASSES)):
        class_indexes = np.flatnonzero(labels == label)
        selected_indexes = random_generator.choice(
            class_indexes,
            size=PCA_SAMPLES_PER_CLASS,
            replace=False,
        )
        pca_indexes.extend(selected_indexes)

    pca_indexes = np.array(pca_indexes)
    random_generator.shuffle(pca_indexes)

    pca_labels = labels[pca_indexes]
    pca_images = images[pca_indexes].reshape(len(pca_indexes), -1)
    pca_images = pca_images.astype(np.float32) / 255.0

    print(f"\nPCA startar med {len(pca_images)} bilder.")

    # --------------------------------------------------------
    # 1.3 Anpassa PCA
    # --------------------------------------------------------
    try:
        pca = PCA(n_components=PCA_VARIANCE)
        pca_result = pca.fit_transform(pca_images)
    except ValueError as error:
        print(f"\nPCA kunde inte genomföras: {error}")
        return False

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    component_numbers = np.arange(1, len(explained_variance) + 1)

    pca_animals = np.array([ANIMAL_CLASSES[int(label)] for label in pca_labels])
    pca_groups = np.array([animal_to_group[animal] for animal in pca_animals])

    pca_coordinates = pd.DataFrame(
        {
            "pca_1": pca_result[:, 0],
            "pca_2": pca_result[:, 1],
            "label": pca_labels,
            "animal": pca_animals,
            "animal_group": pca_groups,
        }
    )
    variance_table = pd.DataFrame(
        {
            "component": component_numbers,
            "explained_variance_ratio": explained_variance,
            "cumulative_explained_variance": cumulative_variance,
        }
    )

    # --------------------------------------------------------
    # 1.4 Spara PCA-resultatet
    # --------------------------------------------------------
    PCA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pca_summary_lines = [
        "PCA - TRÄNINGSDATA",
        "========================================",
        f"Antal bilder: {len(pca_images)}",
        f"Bilder per klass: {PCA_SAMPLES_PER_CLASS}",
        f"Antal ursprungliga pixlar: {pca_images.shape[1]}",
        f"Antal PCA-komponenter: {pca_result.shape[1]}",
        f"Bevarad varians: {cumulative_variance[-1]:.4f}",
    ]
    pca_summary = "\n".join(pca_summary_lines)

    (PCA_OUTPUT_DIR / "summary.txt").write_text(pca_summary, encoding="utf-8")
    variance_table.to_csv(PCA_OUTPUT_DIR / "explained_variance.csv", index=False)
    pca_coordinates.to_csv(PCA_OUTPUT_DIR / "coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(component_numbers, cumulative_variance, color="#4C78A8")
    ax.axhline(PCA_VARIANCE, color="#E45756", linestyle="--")
    ax.axvline(len(explained_variance), color="#E45756", linestyle="--")
    ax.set_title("Kumulativ förklarad varians för PCA")
    ax.set_xlabel("Antal komponenter")
    ax.set_ylabel("Kumulativ förklarad varians")
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(PCA_OUTPUT_DIR / "explained_variance.png", dpi=150)
    plt.close(fig)

    _save_projection(
        coordinates=pca_result[:, :2],
        categories=pca_animals,
        category_order=ANIMAL_CLASSES,
        title="PCA-projektion färgad efter djurklass",
        x_label=f"PCA 1 ({explained_variance[0]:.1%})",
        y_label=f"PCA 2 ({explained_variance[1]:.1%})",
        output_path=PCA_OUTPUT_DIR / "projection_by_class.png",
    )
    _save_projection(
        coordinates=pca_result[:, :2],
        categories=pca_groups,
        category_order=list(ANIMAL_GROUPS),
        title="PCA-projektion färgad efter djurgrupp",
        x_label=f"PCA 1 ({explained_variance[0]:.1%})",
        y_label=f"PCA 2 ({explained_variance[1]:.1%})",
        output_path=PCA_OUTPUT_DIR / "projection_by_group.png",
    )

    print(f"PCA är klar. Resultatet har sparats i {PCA_OUTPUT_DIR}.")

    # --------------------------------------------------------
    # 1.5 Välj PCA-representationer för UMAP
    # --------------------------------------------------------
    umap_positions = []

    for label in range(len(ANIMAL_CLASSES)):
        class_positions = np.flatnonzero(pca_labels == label)
        selected_positions = random_generator.choice(
            class_positions,
            size=UMAP_SAMPLES_PER_CLASS,
            replace=False,
        )
        umap_positions.extend(selected_positions)

    umap_positions = np.array(umap_positions)
    random_generator.shuffle(umap_positions)

    umap_input = pca_result[umap_positions]
    umap_labels = pca_labels[umap_positions]
    umap_animals = pca_animals[umap_positions]
    umap_groups = pca_groups[umap_positions]

    print(f"UMAP startar med {len(umap_input)} bilder.")

    # --------------------------------------------------------
    # 1.6 Anpassa UMAP
    # --------------------------------------------------------
    try:
        umap_model = UMAP(
            n_components=2,
            n_neighbors=UMAP_N_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            metric="euclidean",
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
        umap_result = umap_model.fit_transform(umap_input)
    except (ValueError, RuntimeError) as error:
        print(f"\nUMAP kunde inte genomföras: {error}")
        return False

    umap_coordinates = pd.DataFrame(
        {
            "umap_1": umap_result[:, 0],
            "umap_2": umap_result[:, 1],
            "label": umap_labels,
            "animal": umap_animals,
            "animal_group": umap_groups,
        }
    )

    # --------------------------------------------------------
    # 1.7 Spara UMAP-resultatet
    # --------------------------------------------------------
    UMAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    umap_summary_lines = [
        "UMAP - TRÄNINGSDATA",
        "========================================",
        f"Antal bilder: {len(umap_input)}",
        f"Bilder per klass: {UMAP_SAMPLES_PER_CLASS}",
        f"Antal PCA-komponenter som indata: {umap_input.shape[1]}",
        f"n_neighbors: {UMAP_N_NEIGHBORS}",
        f"min_dist: {UMAP_MIN_DIST}",
        "Mått: euclidean",
    ]
    umap_summary = "\n".join(umap_summary_lines)

    (UMAP_OUTPUT_DIR / "summary.txt").write_text(umap_summary, encoding="utf-8")
    umap_coordinates.to_csv(UMAP_OUTPUT_DIR / "coordinates.csv", index=False)

    _save_projection(
        coordinates=umap_result,
        categories=umap_animals,
        category_order=ANIMAL_CLASSES,
        title="UMAP-projektion färgad efter djurklass",
        x_label="UMAP 1",
        y_label="UMAP 2",
        output_path=UMAP_OUTPUT_DIR / "projection_by_class.png",
    )
    _save_projection(
        coordinates=umap_result,
        categories=umap_groups,
        category_order=list(ANIMAL_GROUPS),
        title="UMAP-projektion färgad efter djurgrupp",
        x_label="UMAP 1",
        y_label="UMAP 2",
        output_path=UMAP_OUTPUT_DIR / "projection_by_group.png",
    )

    print(f"UMAP är klar. Resultatet har sparats i {UMAP_OUTPUT_DIR}.")
    print(f"\n{pca_summary}\n\n{umap_summary}")

    return True


# ============================================================
# 2. SPARA EN PROJEKTION
# ============================================================
# Samma figurformat används för att kunna jämföra PCA och UMAP.
def _save_projection(
    coordinates: np.ndarray,
    categories: np.ndarray,
    category_order: list[str],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 10))

    if len(category_order) <= 10:
        colors = list(plt.get_cmap("tab10").colors)
    else:
        colors = (
            list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors)
        )

    for index, category in enumerate(category_order):
        category_mask = categories == category
        ax.scatter(
            coordinates[category_mask, 0],
            coordinates[category_mask, 1],
            s=8,
            alpha=0.55,
            color=colors[index],
            label=category,
        )

    legend_columns = 4 if len(category_order) > 10 else len(category_order)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=legend_columns,
        fontsize=8,
        markerscale=2,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 3. STARTA PCA OCH UMAP
# ============================================================
# Funktionen kan köras direkt eller genom projektets pipeline-meny.
if __name__ == "__main__":
    run_pca_umap()
