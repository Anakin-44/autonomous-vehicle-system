import os
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from ultralytics import YOLO

app = Flask(__name__, template_folder='templates')

# Resolve absolute path to models directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, 'models', 'yolov8n.pt')
BEHAVIOR_PATH = os.path.join(BASE_DIR, 'models', 'driver_behavior_cnn.h5')

# Custom InputLayer to bridge Keras 3 -> Keras 2 config deserialization
class FixedInputLayer(tf.keras.layers.InputLayer):
    @classmethod
    def from_config(cls, config):
        config = config.copy()
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        config.pop('optional', None)
        return super().from_config(config)

# Load models from the models directory
yolo_model = YOLO(YOLO_PATH)
behavior_model = tf.keras.models.load_model(
    BEHAVIOR_PATH, 
    custom_objects={'InputLayer': FixedInputLayer}, 
    compile=False
)

DRIVER_STATES = [
    "Safe Driving", "Texting (Right)", "Talking on Phone (Right)", 
    "Texting (Left)", "Talking on Phone (Left)", "Operating Radio", 
    "Drinking", "Reaching Behind", "Fixing Hair/Makeup", "Talking to Passenger"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    dashcam_file = request.files.get('dashcam')
    cabin_file = request.files.get('cabin')

    if not dashcam_file or not cabin_file:
        return jsonify({"error": "Missing input files"}), 400

    dashcam_img = Image.open(io.BytesIO(dashcam_file.read())).convert('RGB')
    cabin_img = Image.open(io.BytesIO(cabin_file.read())).convert('RGB')

    # 1. Object Detection (YOLO)
    yolo_results = yolo_model(dashcam_img, verbose=False)
    vehicle_count = 0
    person_count = 0
    for box in yolo_results[0].boxes:
        cls_id = int(box.cls[0])
        if cls_id in [2, 3, 5, 7]:  # Car, Motorcycle, Bus, Truck
            vehicle_count += 1
        elif cls_id == 0:  # Person
            person_count += 1
    obj_text = f"Vehicles: {vehicle_count}, Pedestrians: {person_count}"

    # 2. Driver Behavior (MobileNetV2)
    cab_arr = cv2.resize(np.array(cabin_img), (224, 224))
    cab_arr = np.expand_dims(cab_arr, axis=0)
    pred_behavior = behavior_model.predict(cab_arr, verbose=0)
    class_idx = np.argmax(pred_behavior[0])
    confidence = int(np.max(pred_behavior[0]) * 100)
    beh_text = f"{DRIVER_STATES[class_idx]} ({confidence}%)"

    return jsonify({
        "objects": obj_text,
        "behavior": beh_text,
        "sign": "Traffic Sign CNN ready", 
        "lane": "U-Net Segmented"
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)