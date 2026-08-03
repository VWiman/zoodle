"""Run the public drawing game for the Zoodle project."""

from html import escape
from io import BytesIO
from pathlib import Path
import random
import sys

import streamlit as st
from PIL import Image

# Projektroten läggs till så att appen hittar settings både lokalt och i cloud.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from app.drawing_canvas import drawing_canvas
from app.inference import (
    MAX_IMAGE_DIMENSION,
    load_cnn_model,
    predict_doodle,
    prepare_doodle_image,
)
from settings import ANIMAL_CLASSES, ANIMAL_NAMES_SV


APP_DIR = Path(__file__).resolve().parent
CSS_PATH = APP_DIR / "assets" / "zoodle.css"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


# ============================================================
# 1. LADDA MODELLEN
# ============================================================
# Modellen laddas en gång och återanvänds sedan under hela appsessionen.
@st.cache_resource(show_spinner=False)
def get_model():
    return load_cnn_model()


# ============================================================
# 2. HANTERA EN SPELOMGÅNG
# ============================================================
# Måldjuret sparas i session state så att det inte ändras vid varje rerun.
def initialize_session() -> None:
    if "target_animal" not in st.session_state:
        st.session_state.target_animal = random.choice(ANIMAL_CLASSES)

    if "input_reset_token" not in st.session_state:
        st.session_state.input_reset_token = 0

    if "predictions" not in st.session_state:
        st.session_state.predictions = None

    if "prepared_image" not in st.session_state:
        st.session_state.prepared_image = None

    if "prediction_error" not in st.session_state:
        st.session_state.prediction_error = None


# ------------------------------------------------------------
# 2.1 Ta bort ett tidigare resultat
# ------------------------------------------------------------
# Resultatet rensas när användaren väljer en annan bild.
def clear_result() -> None:
    st.session_state.predictions = None
    st.session_state.prepared_image = None
    st.session_state.prediction_error = None


# ------------------------------------------------------------
# 2.2 Återställ den aktuella teckningen
# ------------------------------------------------------------
# Samma djur behålls, men gammal bild och tidigare resultat tas bort.
def reset_attempt() -> None:
    st.session_state.input_reset_token += 1
    clear_result()


# ------------------------------------------------------------
# 2.3 Välj ett nytt djur
# ------------------------------------------------------------
# Det nya djuret väljs bland de övriga klasserna så att det alltid byts ut.
def choose_next_animal() -> None:
    current_animal = st.session_state.target_animal
    available_animals = [
        animal for animal in ANIMAL_CLASSES if animal != current_animal
    ]
    st.session_state.target_animal = random.choice(available_animals)
    reset_attempt()


# ============================================================
# 3. KLASSIFICERA EN BILD
# ============================================================
# Ritad och uppladdad bild skickas genom exakt samma preprocessing och modell.
def classify_image(image: Image.Image) -> None:
    st.session_state.prediction_error = None

    try:
        with st.spinner("Zoodle funderar..."):
            prepared_image = prepare_doodle_image(image)
            predictions = predict_doodle(
                get_model(),
                prepared_image,
                top_k=3,
            )
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        st.session_state.predictions = None
        st.session_state.prepared_image = None
        st.session_state.prediction_error = str(error)
        return

    st.session_state.prepared_image = prepared_image
    st.session_state.predictions = predictions


# ============================================================
# 4. VISA SPELETS DELAR
# ============================================================
# Gränssnittet delas upp i små funktioner för att huvudflödet ska vara lättläst.
def show_header() -> None:
    st.html(
        """
        <p class="zoodle-eyebrow">Ett ritspel med maskininlärning</p>
        <h1>Zoodle</h1>
        <p class="zoodle-intro">
            Rita djuret som visas eller ladda upp en egen doodle.
            Sedan får Zoodle försöka lista ut vad bilden föreställer.
        </p>
        """
    )


def show_challenge() -> None:
    animal_name = ANIMAL_NAMES_SV[st.session_state.target_animal].capitalize()
    st.html(
        f"""
        <div class="zoodle-challenge">
            <span class="zoodle-challenge__label">Ditt djur</span>
            <span class="zoodle-challenge__animal">{escape(animal_name)}</span>
        </div>
        """
    )


# ------------------------------------------------------------
# 4.1 Visa ritytan
# ------------------------------------------------------------
# Canvasen returnerar en PNG-bild först när användaren ber modellen gissa.
def show_drawing_input() -> None:
    image_bytes = drawing_canvas(
        key="zoodle_canvas",
        reset_token=st.session_state.input_reset_token,
    )

    if image_bytes is None:
        return

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            classify_image(image.copy())
    except OSError:
        st.session_state.prediction_error = "Den ritade bilden kunde inte läsas."


# ------------------------------------------------------------
# 4.2 Visa bilduppladdningen
# ------------------------------------------------------------
# Bilden läses endast i minnet och sparas aldrig i projektet.
def show_upload_input() -> None:
    uploaded_file = st.file_uploader(
        "Ladda upp en doodle",
        type=["png", "jpg", "jpeg"],
        key=f"zoodle_upload_{st.session_state.input_reset_token}",
        help="Använd en tydlig doodle med enkel bakgrund. Maximal filstorlek är 5 MB.",
        on_change=clear_result,
    )

    if uploaded_file is None:
        return

    uploaded_bytes = uploaded_file.getvalue()
    if len(uploaded_bytes) > MAX_UPLOAD_BYTES:
        st.error("Bilden får vara högst 5 MB.")
        return

    try:
        with Image.open(BytesIO(uploaded_bytes)) as image:
            if max(image.size) > MAX_IMAGE_DIMENSION:
                st.error(
                    f"Bilden får vara högst {MAX_IMAGE_DIMENSION} pixlar "
                    "på längsta sidan."
                )
                return

            image.load()
            uploaded_image = image.copy()
    except (OSError, Image.DecompressionBombError):
        st.error("Bilden kunde inte läsas. Välj en giltig PNG- eller JPEG-fil.")
        return

    st.image(uploaded_image, caption="Din uppladdade doodle", width="stretch")

    if st.button(
        "Låt Zoodle gissa",
        type="primary",
        width="stretch",
    ):
        classify_image(uploaded_image)


# ------------------------------------------------------------
# 4.3 Visa modellens resultat
# ------------------------------------------------------------
# Resultatet visar top-1 som svar och de tre högsta sannolikheterna som stöd.
def show_result() -> None:
    if st.session_state.prediction_error:
        st.error(st.session_state.prediction_error)

    predictions = st.session_state.predictions
    if not predictions:
        return

    target_animal = st.session_state.target_animal
    target_name = ANIMAL_NAMES_SV[target_animal].capitalize()
    predicted_animal = predictions[0]["animal"]
    predicted_name = ANIMAL_NAMES_SV[predicted_animal].capitalize()
    is_correct = predicted_animal == target_animal
    target_rank = next(
        (
            rank
            for rank, prediction in enumerate(predictions, start=1)
            if prediction["animal"] == target_animal
        ),
        None,
    )

    if is_correct:
        result_class = "zoodle-result--correct"
        result_kicker = "Rätt!"
        result_title = f"Zoodle gissade {predicted_name}"
        result_text = "Teckningen matchade djuret i utmaningen."
    else:
        result_class = "zoodle-result--wrong"
        result_kicker = "Inte riktigt"
        result_title = f"Zoodle gissade {predicted_name}"
        result_text = f"Djuret i utmaningen var {target_name}."

    near_text = ""
    if target_rank in (2, 3):
        near_text = (
            '<span class="zoodle-near">'
            f"Nära! {escape(target_name)} fanns på plats {target_rank}."
            "</span>"
        )

    st.html(
        f"""
        <div class="zoodle-result {result_class}">
            <p class="zoodle-result__kicker">{escape(result_kicker)}</p>
            <p class="zoodle-result__title">{escape(result_title)}</p>
            <p class="zoodle-result__text">{escape(result_text)}</p>
            {near_text}
        </div>
        """
    )

    show_predictions(predictions)

    with st.expander("Så här såg modellen bilden"):
        prepared_image = st.session_state.prepared_image[0, :, :, 0]
        st.image(
            prepared_image,
            caption="Förbehandlad bild i storleken 28 × 28 pixlar",
            width=220,
            clamp=True,
        )

    retry_column, next_column = st.columns(2)
    retry_column.button(
        "Försök igen",
        on_click=reset_attempt,
        width="stretch",
    )
    next_column.button(
        "Nästa djur",
        type="primary",
        on_click=choose_next_animal,
        width="stretch",
    )


def show_predictions(predictions: list[dict]) -> None:
    prediction_rows = []

    for prediction in predictions:
        animal_name = ANIMAL_NAMES_SV[prediction["animal"]].capitalize()
        confidence = float(prediction["confidence"])
        percentage = confidence * 100
        bar_width = min(max(percentage, 0), 100)
        prediction_rows.append(
            f"""
            <div class="zoodle-prediction">
                <div class="zoodle-prediction__header">
                    <span class="zoodle-prediction__label">{escape(animal_name)}</span>
                    <span class="zoodle-prediction__value">{percentage:.1f} %</span>
                </div>
                <div class="zoodle-prediction__track">
                    <div class="zoodle-prediction__fill" style="width: {bar_width:.1f}%"></div>
                </div>
            </div>
            """
        )

    st.html(
        '<div class="zoodle-predictions">'
        '<p class="zoodle-result__kicker">Zoodles tre främsta gissningar</p>'
        f"{''.join(prediction_rows)}"
        "</div>"
    )


# ------------------------------------------------------------
# 4.4 Visa information om projektet
# ------------------------------------------------------------
# Projektinformationen hålls kort så att spelet förblir sidans huvudfokus.
def show_about() -> None:
    st.html(
        """
        <div class="zoodle-about">
            <strong>Om Zoodle</strong>
            <p>
                Zoodle använder en CNN som har tränats på 48 djurklasser från
                Googles <a href="https://quickdraw.withgoogle.com/data" target="_blank">
                Quick, Draw!-dataset</a>. Modellen nådde 68,12 procent accuracy
                på projektets testdata. Bilder som ritas eller laddas upp sparas inte.
            </p>
            <a href="https://www.viktorwiman.se" target="_blank">viktorwiman.se</a>
        </div>
        """
    )


# ============================================================
# 5. STARTA APPEN
# ============================================================
# Huvudflödet visar en inputtyp i taget och därefter det senaste resultatet.
def main() -> None:
    st.set_page_config(
        page_title="Zoodle",
        page_icon="✏️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.html(CSS_PATH)
    initialize_session()

    show_header()
    show_challenge()

    input_mode = st.segmented_control(
        "Välj hur du vill skapa bilden",
        ["Rita själv", "Ladda upp"],
        default="Rita själv",
        key="input_mode",
        on_change=reset_attempt,
        label_visibility="collapsed",
        width="stretch",
    )

    if input_mode == "Ladda upp":
        show_upload_input()
    else:
        show_drawing_input()

    show_result()
    show_about()


if __name__ == "__main__":
    main()
