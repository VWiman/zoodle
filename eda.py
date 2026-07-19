"""Explore the prepared training data used by the Zoodle project."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from settings import (
    ANIMAL_CLASSES,
    EDA_OUTPUT_DIR,
    IMAGE_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
)


# ============================================================
# 1. UTFORSKA TRÄNINGSDATAN
# ============================================================
# Endast träningsdatan används så att validerings- och testdata förblir orörda.
def run_eda() -> bool:
    train_path = PROCESSED_DATA_DIR / "train.npz"

    if not train_path.exists():
        print(f"\nTräningsdatan saknas: {train_path}")
        print("Kör dataförberedelsen innan EDA startas.")
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

    # --------------------------------------------------------
    # 1.2 Beräkna statistik
    # --------------------------------------------------------
    flat_images = images.reshape(len(images), -1)
    ink_pixels = np.count_nonzero(flat_images, axis=1)
    empty_images = int((ink_pixels == 0).sum())
    class_counts = np.bincount(labels, minlength=len(ANIMAL_CLASSES))

    if np.any(class_counts == 0):
        missing_classes = [
            animal
            for label, animal in enumerate(ANIMAL_CLASSES)
            if class_counts[label] == 0
        ]
        print(f"\nTräningsdatan saknar följande klasser: {missing_classes}")
        return False

    class_statistics = []

    for label, animal in enumerate(ANIMAL_CLASSES):
        class_ink = ink_pixels[labels == label]
        class_statistics.append(
            {
                "label": label,
                "animal": animal,
                "image_count": int(class_counts[label]),
                "average_ink_pixels": float(class_ink.mean()),
                "median_ink_pixels": float(np.median(class_ink)),
            }
        )

    statistics_df = pd.DataFrame(class_statistics)

    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1.3 Spara sammanfattning och klasstatistik
    # --------------------------------------------------------
    summary_lines = [
        "EDA - TRÄNINGSDATA",
        "========================================",
        f"Antal bilder: {len(images)}",
        f"Antal klasser: {len(ANIMAL_CLASSES)}",
        f"Bildform: {images.shape[1:]}",
        f"Datatyp: {images.dtype}",
        f"Minsta pixelvärde: {int(images.min())}",
        f"Största pixelvärde: {int(images.max())}",
        f"Tomma bilder: {empty_images}",
        f"Minsta antal bilder per klass: {int(class_counts.min())}",
        f"Största antal bilder per klass: {int(class_counts.max())}",
        f"Genomsnittligt antal pixlar med bläck: {ink_pixels.mean():.2f}",
        f"Medianantal pixlar med bläck: {np.median(ink_pixels):.2f}",
    ]

    summary_text = "\n".join(summary_lines)
    (EDA_OUTPUT_DIR / "summary.txt").write_text(summary_text, encoding="utf-8")
    statistics_df.to_csv(EDA_OUTPUT_DIR / "class_statistics.csv", index=False)

    # --------------------------------------------------------
    # 1.4 Spara klassfördelningen
    # --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(ANIMAL_CLASSES, class_counts, color="#4C78A8")
    ax.set_title("Antal träningsbilder per djurklass")
    ax.set_xlabel("Djurklass")
    ax.set_ylabel("Antal bilder")
    ax.tick_params(axis="x", rotation=90)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # 1.5 Spara en exempelbild per klass
    # --------------------------------------------------------
    random_generator = np.random.default_rng(RANDOM_STATE)
    fig, axes = plt.subplots(6, 8, figsize=(16, 12))

    for label, ax in enumerate(axes.flat):
        class_indexes = np.flatnonzero(labels == label)
        image_index = random_generator.choice(class_indexes)
        ax.imshow(images[image_index].squeeze(), cmap="gray", vmin=0, vmax=255)
        ax.set_title(ANIMAL_CLASSES[label], fontsize=10)
        ax.axis("off")

    fig.suptitle("Exempelbild från varje djurklass", fontsize=16)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "sample_images.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # 1.6 Spara medelbilden för varje klass
    # --------------------------------------------------------
    fig, axes = plt.subplots(6, 8, figsize=(16, 12))

    for label, ax in enumerate(axes.flat):
        average_image = images[labels == label].mean(axis=0).squeeze()
        ax.imshow(average_image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(ANIMAL_CLASSES[label], fontsize=10)
        ax.axis("off")

    fig.suptitle("Medelbild för varje djurklass", fontsize=16)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "average_images.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # 1.7 Spara genomsnittlig bläckmängd per klass
    # --------------------------------------------------------
    sorted_statistics = statistics_df.sort_values("average_ink_pixels")
    fig, ax = plt.subplots(figsize=(11, 12))
    ax.barh(
        sorted_statistics["animal"],
        sorted_statistics["average_ink_pixels"],
        color="#54A24B",
    )
    ax.set_title("Genomsnittligt antal pixlar med bläck per djurklass")
    ax.set_xlabel("Antal pixlar med bläck")
    ax.set_ylabel("Djurklass")
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "average_ink_per_class.png", dpi=150)
    plt.close(fig)

    # --------------------------------------------------------
    # 1.8 Spara fördelningen av pixelvärden
    # --------------------------------------------------------
    pixel_counts = np.bincount(flat_images.ravel(), minlength=256)
    pixel_values = np.arange(1, 256)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(pixel_values, pixel_counts[1:], color="#F58518")
    ax.set_title("Fördelning av pixelvärden över noll")
    ax.set_xlabel("Pixelvärde")
    ax.set_ylabel("Antal pixlar")
    ax.set_xlim(1, 255)
    fig.tight_layout()
    fig.savefig(EDA_OUTPUT_DIR / "pixel_intensity_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\n{summary_text}")
    print(f"\nEDA är klar. Resultatet har sparats i {EDA_OUTPUT_DIR}.")

    return True


# ============================================================
# 2. STARTA EDA
# ============================================================
# Funktionen kan köras direkt eller genom projektets pipeline-meny.
if __name__ == "__main__":
    run_eda()
