
import os
import io
import logging

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO


# --------------------------------------------------
# 1. Flask Configuration
# --------------------------------------------------

app = Flask(__name__, template_folder="templates")

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB per request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# 2. Model Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
BEHAVIOR_PATH = os.path.join(
    BASE_DIR, "models", "driver_behavior_cnn.h5"
)


# --------------------------------------------------
# 3. Driver Behavior Classes
# --------------------------------------------------

DRIVER_STATES = [
    "Safe Driving",
    "Texting (Right)",
    "Talking on Phone (Right)",
    "Texting (Left)",
    "Talking on Phone (Left)",
    "Operating Radio",
    "Drinking",
    "Reaching Behind",
    "Fixing Hair/Makeup",
    "Talking to Passenger",
]


# --------------------------------------------------
# 4. Load Deep Learning Models
# --------------------------------------------------

def load_models():
    """Load the YOLO and driver behavior models."""

    if not os.path.isfile(YOLO_PATH):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_PATH}"
        )

    if not os.path.isfile(BEHAVIOR_PATH):
        raise FileNotFoundError(
            f"Driver behavior model not found: {BEHAVIOR_PATH}"
        )

    logger.info("Loading YOLO model...")
    yolo = YOLO(YOLO_PATH)

    logger.info("Loading driver behavior model...")
    behavior = tf.keras.models.load_model(
        BEHAVIOR_PATH,
        compile=False
    )

    # Validate the number of output classes.
    output_shape = behavior.output_shape
    output_classes = output_shape[-1]

    if output_classes != len(DRIVER_STATES):
        raise ValueError(
            f"Driver model outputs {output_classes} classes, "
            f"but DRIVER_STATES contains {len(DRIVER_STATES)}."
        )

    logger.info("All required models loaded successfully!")

    return yolo, behavior


yolo_model, behavior_model = load_models()


# --------------------------------------------------
# 5. Home Page
# --------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# 6. Health Check
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "yolo_loaded": yolo_model is not None,
        "behavior_model_loaded": behavior_model is not None,
    }), 200


# --------------------------------------------------
# 7. Analyze Uploaded Images
# --------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    dashcam_file = request.files.get("dashcam")
    cabin_file = request.files.get("cabin")

    if not dashcam_file or not cabin_file:
        return jsonify({
            "error": "Please upload both dashcam and cabin images."
        }), 400

    try:
        # Read and validate images.
        dashcam_img = Image.open(
            io.BytesIO(dashcam_file.read())
        ).convert("RGB")

        cabin_img = Image.open(
            io.BytesIO(cabin_file.read())
        ).convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify({
            "error": "One or both uploaded files are not valid images."
        }), 400

    try:
        # ------------------------------------------
        # A. Object Detection (YOLOv8)
        # ------------------------------------------

        yolo_results = yolo_model(
            dashcam_img,
            verbose=False
        )

        vehicle_count = 0
        person_count = 0

        for box in yolo_results[0].boxes:
            class_id = int(box.cls[0])

            # COCO class IDs:
            # 0 = person, 2 = car, 3 = motorcycle,
            # 5 = bus, 7 = truck
            if class_id in [2, 3, 5, 7]:
                vehicle_count += 1

            elif class_id == 0:
                person_count += 1

        obj_text = (
            f"Vehicles: {vehicle_count}, "
            f"Pedestrians: {person_count}"
        )

        # ------------------------------------------
        # B. Driver Behavior Classification
        # ------------------------------------------

        cabin_array = np.array(cabin_img)

        cabin_array = cv2.resize(
            cabin_array,
            (224, 224)
        )

        cabin_array = np.expand_dims(
            cabin_array,
            axis=0
        )

        # Keep preprocessing consistent with training.
        predictions = behavior_model.predict(
            cabin_array,
            verbose=0
        )

        class_idx = int(np.argmax(predictions[0]))
        confidence = float(np.max(predictions[0])) * 100

        if not 0 <= class_idx < len(DRIVER_STATES):
            raise ValueError(
                f"Unexpected driver behavior class index: {class_idx}"
            )

        behavior_text = (
            f"{DRIVER_STATES[class_idx]} "
            f"({confidence:.1f}%)"
        )

        # ------------------------------------------
        # C. Response
        # ------------------------------------------

        return jsonify({
            "objects": obj_text,
            "behavior": behavior_text,

            # These are placeholders, not model predictions.
            "sign": "Traffic Sign model integration pending",
            "lane": "Lane Detection model integration pending",
        }), 200

    except Exception:
        logger.exception("Image analysis failed")

        return jsonify({
            "error": "Analysis failed. Please check the server logs."
        }), 500


# --------------------------------------------------
# 8. Request Size Error Handler
# --------------------------------------------------

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "error": "Uploaded files are too large. Maximum request size is 10 MB."
    }), 413


# --------------------------------------------------
# 9. Application Entry Point
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
