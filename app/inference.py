"""Prepare doodle images and run inference with the trained Zoodle CNN."""

from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

from settings import ANIMAL_CLASSES, IMAGE_SIZE


MODEL_PATH = Path(__file__).resolve().parent / "model" / "zoodle_cnn.keras"
CONTENT_SIZE = IMAGE_SIZE - 2
INK_THRESHOLD = 20
MIN_INK_PIXELS = 5
MAX_IMAGE_DIMENSION = 4096


# ============================================================
# 1. LADDA CNN-MODELLEN
# ============================================================
# Modellen laddas utan optimizerstatus eftersom appen endast använder inference.
def load_cnn_model(model_path: Path = MODEL_PATH) -> tf.keras.Model:
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"CNN-modellen saknas: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (OSError, ValueError, tf.errors.OpError) as error:
        raise ValueError(f"CNN-modellen kunde inte läsas: {error}") from error

    expected_input_shape = (None, IMAGE_SIZE, IMAGE_SIZE, 1)
    expected_output_shape = (None, len(ANIMAL_CLASSES))

    if model.input_shape != expected_input_shape:
        raise ValueError(
            f"CNN-modellen har fel indataform: {model.input_shape}. "
            f"Förväntad form är {expected_input_shape}."
        )

    if model.output_shape != expected_output_shape:
        raise ValueError(
            f"CNN-modellen har fel utdataform: {model.output_shape}. "
            f"Förväntad form är {expected_output_shape}."
        )

    return model


# ============================================================
# 2. FÖRBERED EN TECKNING
# ============================================================
# Canvas och uppladdade bilder behandlas på samma sätt innan inference.
def prepare_doodle_image(image: Image.Image | np.ndarray) -> np.ndarray:
    source_image = _to_pillow_image(image)
    source_image = ImageOps.exif_transpose(source_image)

    if max(source_image.size) > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"Bilden får vara högst {MAX_IMAGE_DIMENSION} pixlar på längsta sidan."
        )

    # --------------------------------------------------------
    # 2.1 Skapa en gråskalebild med vit bakgrund
    # --------------------------------------------------------
    # Transparens läggs på vitt så att canvas och uppladdade PNG-filer får
    # samma utgångsläge.
    if "A" in source_image.getbands() or "transparency" in source_image.info:
        rgba_image = source_image.convert("RGBA")
        white_background = Image.new("RGBA", rgba_image.size, "white")
        source_image = Image.alpha_composite(white_background, rgba_image)

    gray_image = source_image.convert("L")
    gray_pixels = np.asarray(gray_image, dtype=np.uint8)

    if gray_pixels.size == 0:
        raise ValueError("Bilden är tom.")

    # --------------------------------------------------------
    # 2.2 Gör strecken ljusa och bakgrunden svart
    # --------------------------------------------------------
    # Quick, Draw!-bilderna har ljusa streck på svart bakgrund. Medianen längs
    # bildens kant används för att även stödja svarta streck på vit bakgrund.
    border_pixels = _get_border_pixels(gray_pixels)
    has_light_background = float(np.median(border_pixels)) >= 128

    if has_light_background:
        ink_pixels = 255.0 - gray_pixels.astype(np.float32)
    else:
        ink_pixels = gray_pixels.astype(np.float32)

    background_level = float(np.median(_get_border_pixels(ink_pixels)))
    ink_pixels = np.clip(ink_pixels - background_level, 0, 255)

    maximum_ink = float(ink_pixels.max())
    if maximum_ink < INK_THRESHOLD:
        raise ValueError("Bilden är tom eller har för låg kontrast.")

    ink_pixels *= 255.0 / maximum_ink
    ink_pixels[ink_pixels < INK_THRESHOLD] = 0
    ink_pixels = np.rint(ink_pixels).astype(np.uint8)

    content_mask = ink_pixels > 0
    if int(content_mask.sum()) < MIN_INK_PIXELS:
        raise ValueError("Bilden innehåller för lite teckning.")

    # --------------------------------------------------------
    # 2.3 Beskär, skala och centrera teckningen
    # --------------------------------------------------------
    # Innehållet får högst 26 x 26 pixlar och centreras på samma svarta
    # 28 x 28-yta som bilderna i träningsdatan.
    row_indexes, column_indexes = np.where(content_mask)
    top = int(row_indexes.min())
    bottom = int(row_indexes.max()) + 1
    left = int(column_indexes.min())
    right = int(column_indexes.max()) + 1

    cropped_pixels = ink_pixels[top:bottom, left:right]
    cropped_height, cropped_width = cropped_pixels.shape
    scale = CONTENT_SIZE / max(cropped_width, cropped_height)
    resized_width = max(1, round(cropped_width * scale))
    resized_height = max(1, round(cropped_height * scale))

    cropped_image = Image.fromarray(cropped_pixels, mode="L")
    resized_image = cropped_image.resize(
        (resized_width, resized_height),
        Image.Resampling.LANCZOS,
    )

    prepared_canvas = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=0)
    paste_left = (IMAGE_SIZE - resized_width) // 2
    paste_top = (IMAGE_SIZE - resized_height) // 2
    prepared_canvas.paste(resized_image, (paste_left, paste_top))

    prepared_pixels = np.asarray(prepared_canvas, dtype=np.uint8)
    return prepared_pixels[np.newaxis, :, :, np.newaxis]


# ============================================================
# 3. KLASSIFICERA EN TECKNING
# ============================================================
# Modellen anropas direkt i eager mode för samma säkra inference som i
# projektets utvärderingsflöde.
def predict_doodle(
    model: tf.keras.Model,
    prepared_image: np.ndarray,
    top_k: int = 3,
) -> list[dict[str, int | str | float]]:
    expected_shape = (1, IMAGE_SIZE, IMAGE_SIZE, 1)

    if prepared_image.shape != expected_shape:
        raise ValueError(
            f"Bilden har fel form: {prepared_image.shape}. "
            f"Förväntad form är {expected_shape}."
        )

    if prepared_image.dtype != np.uint8:
        raise ValueError(
            f"Bilden har fel datatyp: {prepared_image.dtype}. "
            "Förväntad datatyp är uint8."
        )

    if not 1 <= top_k <= len(ANIMAL_CLASSES):
        raise ValueError(
            f"top_k måste vara mellan 1 och {len(ANIMAL_CLASSES)}."
        )

    try:
        probabilities = model(prepared_image, training=False).numpy()
    except (ValueError, tf.errors.OpError) as error:
        raise ValueError(f"Modellen kunde inte klassificera bilden: {error}") from error

    expected_output_shape = (1, len(ANIMAL_CLASSES))
    if probabilities.shape != expected_output_shape:
        raise ValueError(
            f"Modellen gav fel utdataform: {probabilities.shape}. "
            f"Förväntad form är {expected_output_shape}."
        )

    class_probabilities = probabilities[0]
    probability_sum = float(class_probabilities.sum())

    if not np.all(np.isfinite(class_probabilities)) or not np.isclose(
        probability_sum,
        1.0,
        atol=1e-4,
    ):
        raise ValueError("Modellen gav ogiltiga sannolikheter.")

    top_labels = np.argsort(class_probabilities)[::-1][:top_k]

    return [
        {
            "label": int(label),
            "animal": ANIMAL_CLASSES[int(label)],
            "confidence": float(class_probabilities[label]),
        }
        for label in top_labels
    ]


# ============================================================
# 4. HJÄLPFUNKTIONER FÖR BILDER
# ============================================================
# NumPy-bilder från canvas konverteras till samma Pillow-format som uppladdningar.
def _to_pillow_image(image: Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        if max(image.size) > MAX_IMAGE_DIMENSION:
            raise ValueError(
                f"Bilden får vara högst {MAX_IMAGE_DIMENSION} pixlar "
                "på längsta sidan."
            )
        return image.copy()

    if not isinstance(image, np.ndarray):
        raise TypeError("Bilden måste vara en Pillow-bild eller en NumPy-array.")

    if image.size == 0 or image.ndim not in (2, 3):
        raise ValueError("Bilden har ett ogiltigt format.")

    image_array = np.asarray(image)

    if not np.all(np.isfinite(image_array)):
        raise ValueError("Bilden innehåller ogiltiga pixelvärden.")

    if np.issubdtype(image_array.dtype, np.floating):
        if image_array.min() >= 0 and image_array.max() <= 1:
            image_array = image_array * 255

    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    if image_array.ndim == 3 and image_array.shape[2] == 1:
        image_array = image_array[:, :, 0]
    elif image_array.ndim == 3 and image_array.shape[2] not in (3, 4):
        raise ValueError("Bilden måste ha en, tre eller fyra färgkanaler.")

    return Image.fromarray(image_array)


# --------------------------------------------------------
# 4.1 Hämta pixlarna längs bildens kant
# --------------------------------------------------------
# Kantpixlarna används för att uppskatta bakgrundens färg och ljushet.
def _get_border_pixels(image: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            image[0, :],
            image[-1, :],
            image[:, 0],
            image[:, -1],
        )
    )
