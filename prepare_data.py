"""Prepare the Quick, Draw! image data used by the Zoodle project."""

import numpy as np
from sklearn.model_selection import train_test_split

from settings import (
    ANIMAL_CLASSES,
    IMAGE_SIZE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_DIR,
    SAMPLES_PER_CLASS,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


# ============================================================
# 1. FÖRBERED DATASETET
# ============================================================
# Råfilerna kontrolleras innan bilderna väljs och delas upp.
def prepare_dataset():
    split_ratios = np.array(
        [TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO],
        dtype=float,
    )

    if np.any(split_ratios <= 0) or not np.isclose(split_ratios.sum(), 1.0):
        print(
            "\nAndelarna för träning, validering och test måste vara positiva "
            "och tillsammans bli 1."
        )
        return False

    missing_files = []

    for animal in ANIMAL_CLASSES:
        file_path = RAW_DATA_DIR / f"{animal}.npy"
        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print("\nDatasetet är inte komplett. Följande filer saknas:")
        for file_path in missing_files:
            print(f"- {file_path}")
        print("\nKör nedladdningen innan dataförberedelsen startas.")
        return False

    random_generator = np.random.default_rng(RANDOM_STATE)
    selected_images = []
    selected_labels = []
    total_empty_images = 0
    expected_pixels = IMAGE_SIZE * IMAGE_SIZE

    print("\nDataförberedelsen startar.")

    for label, animal in enumerate(ANIMAL_CLASSES):
        file_path = RAW_DATA_DIR / f"{animal}.npy"

        try:
            images = np.load(file_path)
        except (OSError, ValueError) as error:
            print(f"\nKunde inte läsa {file_path}: {error}")
            return False

        # --------------------------------------------------------
        # 1.1 Kontrollera klassens bilder
        # --------------------------------------------------------
        if images.ndim != 2 or images.shape[1] != expected_pixels:
            print(
                f"\nFel form för {animal}: {images.shape}. "
                f"Förväntad form är (antal bilder, {expected_pixels})."
            )
            return False

        if images.dtype != np.uint8:
            print(
                f"\nFel datatyp för {animal}: {images.dtype}. "
                "Förväntad datatyp är uint8."
            )
            return False

        if images.shape[0] == 0 or images.min() < 0 or images.max() > 255:
            print(f"\nOgiltiga pixelvärden hittades för {animal}.")
            return False

        # --------------------------------------------------------
        # 1.2 Ta bort tomma bilder
        # --------------------------------------------------------
        non_empty_mask = np.any(images != 0, axis=1)
        empty_images = int((~non_empty_mask).sum())
        images = images[non_empty_mask]
        total_empty_images += empty_images

        if len(images) < SAMPLES_PER_CLASS:
            print(
                f"\n{animal} har bara {len(images)} giltiga bilder. "
                f"Minst {SAMPLES_PER_CLASS} behövs."
            )
            return False

        # --------------------------------------------------------
        # 1.3 Välj lika många bilder från varje klass
        # --------------------------------------------------------
        selected_indexes = random_generator.choice(
            len(images),
            size=SAMPLES_PER_CLASS,
            replace=False,
        )
        class_images = images[selected_indexes]
        class_labels = np.full(SAMPLES_PER_CLASS, label, dtype=np.uint8)

        selected_images.append(class_images)
        selected_labels.append(class_labels)

        print(
            f"[{label + 1}/{len(ANIMAL_CLASSES)}] {animal}: "
            f"{SAMPLES_PER_CLASS} bilder valda, {empty_images} tomma borttagna."
        )

    # --------------------------------------------------------
    # 1.4 Skapa träning, validering och test
    # --------------------------------------------------------
    images = np.concatenate(selected_images)
    images = images.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
    labels = np.concatenate(selected_labels)

    temporary_ratio = VALIDATION_RATIO + TEST_RATIO

    train_images, temporary_images, train_labels, temporary_labels = train_test_split(
        images,
        labels,
        train_size=TRAIN_RATIO,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    relative_test_ratio = TEST_RATIO / temporary_ratio
    validation_images, test_images, validation_labels, test_labels = train_test_split(
        temporary_images,
        temporary_labels,
        test_size=relative_test_ratio,
        stratify=temporary_labels,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # 1.5 Spara den uppdelade datan
    # --------------------------------------------------------
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        PROCESSED_DATA_DIR / "train.npz",
        images=train_images,
        labels=train_labels,
    )
    np.savez_compressed(
        PROCESSED_DATA_DIR / "validation.npz",
        images=validation_images,
        labels=validation_labels,
    )
    np.savez_compressed(
        PROCESSED_DATA_DIR / "test.npz",
        images=test_images,
        labels=test_labels,
    )

    print("\n========================================")
    print("DATAFÖRBEREDELSEN ÄR KLAR")
    print("========================================")
    print(f"Träningsbilder: {len(train_images)}")
    print(f"Valideringsbilder: {len(validation_images)}")
    print(f"Testbilder: {len(test_images)}")
    print(f"Tomma bilder borttagna: {total_empty_images}")
    print(f"Sparad mapp: {PROCESSED_DATA_DIR}")

    return True


# ============================================================
# 2. STARTA DATAFÖRBEREDELSEN
# ============================================================
# Funktionen kan köras direkt eller genom projektets pipeline-meny.
if __name__ == "__main__":
    prepare_dataset()
