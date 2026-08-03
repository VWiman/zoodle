"""Provide the browser-based drawing canvas used by the Zoodle app."""

import base64
import binascii

import streamlit as st


CANVAS_HTML = """
<div class="zoodle-drawing-canvas">
    <div class="canvas-frame">
        <canvas
            aria-label="Rityta för ditt djur"
            width="600"
            height="600"
        ></canvas>
    </div>
    <p class="canvas-message" role="status" aria-live="polite"></p>
    <div class="canvas-actions">
        <button class="clear-button" type="button">Rensa</button>
        <button class="submit-button" type="button">Låt Zoodle gissa</button>
    </div>
</div>
"""


CANVAS_CSS = """
:host {
    display: block;
    width: 100%;
}

* {
    box-sizing: border-box;
}

.zoodle-drawing-canvas {
    width: 100%;
    color: #252525;
    font-family: var(--st-font, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif);
}

.canvas-frame {
    width: min(100%, 560px, calc(100vh - 295px));
    aspect-ratio: 1;
    margin: 0 auto;
    overflow: hidden;
    background: #FFFFFF;
    border: 2px solid #E4DED2;
    border-radius: 22px;
    box-shadow: 0 12px 30px rgba(37, 37, 37, 0.08);
}

canvas {
    display: block;
    width: 100%;
    height: 100%;
    background: #FFFFFF;
    cursor: crosshair;
    touch-action: none;
    user-select: none;
}

.canvas-message {
    width: min(100%, 560px);
    min-height: 1.3rem;
    margin: 0.4rem auto 0;
    color: #6A655D;
    font-size: 0.9rem;
    line-height: 1.5;
    text-align: center;
}

.canvas-message:empty {
    display: none;
}

.canvas-message[data-kind="error"] {
    color: #A44736;
}

.canvas-actions {
    display: grid;
    grid-template-columns: minmax(7rem, 0.8fr) minmax(11rem, 1.2fr);
    gap: 0.75rem;
    width: min(100%, 560px);
    margin: 0.25rem auto 0;
}

button {
    min-height: 2.75rem;
    padding: 0.6rem 1rem;
    border-radius: 999px;
    font: inherit;
    font-weight: 700;
    cursor: pointer;
    transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
}

button:hover {
    transform: translateY(-1px);
}

button:active {
    transform: translateY(0);
}

button:focus-visible {
    outline: 3px solid rgba(82, 123, 118, 0.3);
    outline-offset: 2px;
}

.clear-button {
    color: #527B76;
    background: #FFFFFF;
    border: 1px solid #B9CBC8;
}

.clear-button:hover {
    background: #F2F7F5;
}

.submit-button {
    color: #FFFFFF;
    background: #527B76;
    border: 1px solid #527B76;
    box-shadow: 0 7px 16px rgba(82, 123, 118, 0.2);
}

.submit-button:hover {
    background: #436A66;
}

@media (max-width: 440px) {
    .canvas-frame {
        border-radius: 18px;
    }

    .canvas-actions {
        grid-template-columns: 1fr;
    }
}

@media (prefers-reduced-motion: reduce) {
    button {
        transition: none;
    }
}
"""


CANVAS_JAVASCRIPT = """
export default function(component) {
    const { data, parentElement, setTriggerValue } = component;
    const root = parentElement.querySelector(".zoodle-drawing-canvas");
    const frame = root.querySelector(".canvas-frame");
    const canvas = root.querySelector("canvas");
    const clearButton = root.querySelector(".clear-button");
    const submitButton = root.querySelector(".submit-button");
    const message = root.querySelector(".canvas-message");
    const context = canvas.getContext("2d");

    let isDrawing = false;
    let activePointer = null;

    function configureContext(size) {
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
        context.strokeStyle = "#111111";
        context.fillStyle = "#111111";
        context.lineCap = "round";
        context.lineJoin = "round";
        context.lineWidth = Math.max(10, size * 0.028);
        canvas.dataset.pixelRatio = String(pixelRatio);
    }

    function fillWhite() {
        context.save();
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.fillStyle = "#FFFFFF";
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.restore();
    }

    function resizeCanvas() {
        const size = Math.round(frame.getBoundingClientRect().width);
        const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
        const targetSize = Math.round(size * pixelRatio);

        if (size < 1) {
            return;
        }

        if (canvas.width === targetSize && canvas.height === targetSize) {
            configureContext(size);
            return;
        }

        const previousCanvas = document.createElement("canvas");
        previousCanvas.width = canvas.width;
        previousCanvas.height = canvas.height;
        previousCanvas.getContext("2d").drawImage(canvas, 0, 0);
        const hadDrawing = canvas.dataset.hasDrawing === "true";

        canvas.width = targetSize;
        canvas.height = targetSize;
        configureContext(size);
        fillWhite();

        if (hadDrawing) {
            context.drawImage(
                previousCanvas,
                0,
                0,
                previousCanvas.width,
                previousCanvas.height,
                0,
                0,
                size,
                size,
            );
        }
    }

    function clearCanvas(showMessage = true) {
        fillWhite();
        canvas.dataset.hasDrawing = "false";
        message.dataset.kind = "";
        message.textContent = showMessage ? "Ritytan är rensad." : "";
    }

    function getPoint(event) {
        const bounds = canvas.getBoundingClientRect();
        return {
            x: event.clientX - bounds.left,
            y: event.clientY - bounds.top,
        };
    }

    function startDrawing(event) {
        if (event.pointerType === "mouse" && event.button !== 0) {
            return;
        }

        event.preventDefault();
        isDrawing = true;
        activePointer = event.pointerId;
        canvas.setPointerCapture(event.pointerId);

        const point = getPoint(event);
        context.beginPath();
        context.arc(point.x, point.y, context.lineWidth / 2, 0, Math.PI * 2);
        context.fill();
        context.beginPath();
        context.moveTo(point.x, point.y);
        canvas.dataset.hasDrawing = "true";
        message.textContent = "";
        message.dataset.kind = "";
    }

    function continueDrawing(event) {
        if (!isDrawing || event.pointerId !== activePointer) {
            return;
        }

        event.preventDefault();
        const point = getPoint(event);
        context.lineTo(point.x, point.y);
        context.stroke();
    }

    function stopDrawing(event) {
        if (event.pointerId !== activePointer) {
            return;
        }

        isDrawing = false;
        activePointer = null;
        context.closePath();

        if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
        }
    }

    function submitDrawing() {
        if (canvas.dataset.hasDrawing !== "true") {
            message.textContent = "Rita något innan Zoodle får gissa.";
            message.dataset.kind = "error";
            return;
        }

        message.textContent = "";
        message.dataset.kind = "";
        setTriggerValue("image", canvas.toDataURL("image/png"));
    }

    resizeCanvas();

    const resetToken = String(data?.resetToken ?? "");
    if (root.dataset.resetToken !== resetToken) {
        root.dataset.resetToken = resetToken;
        clearCanvas(false);
    } else {
        message.textContent = "";
        message.dataset.kind = "";
    }

    const resizeObserver = new ResizeObserver(resizeCanvas);
    resizeObserver.observe(frame);

    canvas.addEventListener("pointerdown", startDrawing);
    canvas.addEventListener("pointermove", continueDrawing);
    canvas.addEventListener("pointerup", stopDrawing);
    canvas.addEventListener("pointercancel", stopDrawing);
    clearButton.addEventListener("click", clearCanvas);
    submitButton.addEventListener("click", submitDrawing);

    return () => {
        resizeObserver.disconnect();
        canvas.removeEventListener("pointerdown", startDrawing);
        canvas.removeEventListener("pointermove", continueDrawing);
        canvas.removeEventListener("pointerup", stopDrawing);
        canvas.removeEventListener("pointercancel", stopDrawing);
        clearButton.removeEventListener("click", clearCanvas);
        submitButton.removeEventListener("click", submitDrawing);
    };
}
"""


_DRAWING_CANVAS = st.components.v2.component(
    "zoodle_drawing_canvas",
    html=CANVAS_HTML,
    css=CANVAS_CSS,
    js=CANVAS_JAVASCRIPT,
)


# ============================================================
# 1. VISA RITYTAN
# ============================================================
# PNG-bilden skickas till Python först när användaren ber Zoodle gissa.
def drawing_canvas(key: str, reset_token: int) -> bytes | None:
    result = _DRAWING_CANVAS(
        key=key,
        data={"resetToken": reset_token},
        on_image_change=lambda: None,
    )
    image_data = result.image

    if not isinstance(image_data, str) or not image_data.startswith(
        "data:image/png;base64,"
    ):
        return None

    try:
        return base64.b64decode(image_data.split(",", maxsplit=1)[1], validate=True)
    except (binascii.Error, ValueError):
        return None
