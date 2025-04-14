import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from functools import lru_cache

from src.get_data import read_params
from keras.models import load_model
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import cv2
import base64

config = read_params("params.yaml")

model_path = config['model']['sav_dir']
class_names = list(config['raw_data']['classes'])

@lru_cache()
def get_model():
    return load_model(model_path)

model = get_model()

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(image_bytes):
    image_array = preprocess_image(image_bytes)
    probs = model.predict(image_array)[0]
    predicted_class_idx = np.argmax(probs)

    label_map = {
        "AD": "Alzheimer's Disease",
        "CN": "Cognitively Normal",
        "EMCI": "Early Mild Cognitive Impairment",
        "LMCI": "Late Mild Cognitive Impairment"
    }

    short_class = class_names[predicted_class_idx]
    full_class = label_map.get(short_class, short_class)
    confidence = float(np.max(probs) * 100)

    # gradcam = generate_gradcam(image_array, predicted_class_idx)

    return {
        "predicted_class": full_class,
        "class_code": short_class,
        "confidence": round(confidence, 2),
        # "gradcam": gradcam
    }


def generate_gradcam(img_array, class_index):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(index=-1).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]
    if len(grads.shape) == 3:  # (H, W, Channels)
        weights = tf.reduce_mean(grads, axis=(0, 1))
    elif len(grads.shape) == 1:  # fallback (just in case)
        weights = grads
    else:
        raise ValueError(f"Unexpected grads shape: {grads.shape}")
    cam = np.dot(conv_outputs, weights.numpy())

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    img = np.uint8(img_array[0] * 255)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    pil_img = Image.fromarray(superimposed_img)

    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')

    return encoded_img