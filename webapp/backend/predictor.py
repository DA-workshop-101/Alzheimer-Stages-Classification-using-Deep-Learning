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
    model = load_model(model_path)
    # dummy_input = np.zeros((1, 224, 224, 3))
    # model.predict(dummy_input)
    return model

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

    gradcam = generate_gradcam(image_array, predicted_class_idx)

    print(model.summary())

    return {
        "predicted_class": full_class,
        "class_code": short_class,
        "confidence": round(confidence, 2),
        "gradcam": gradcam
    }


import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

def generate_gradcam(img_array, class_index):
    """
    Robust GradCAM implementation that handles:
    - Proper layer selection
    - Tensor/NumPy conversion
    - Image type conversion
    - Dimension validation
    - Error handling
    """
    try:
        # 1. Get the last convolutional layer
        last_conv_layer = model.get_layer('block5_conv4')
        
        # 2. Create gradient model
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[last_conv_layer.output, model.output]
        )

        # 3. Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_index]
        
        # 4. Get gradients and weights
        grads = tape.gradient(loss, conv_outputs)[0].numpy()
        weights = np.mean(grads, axis=(0, 1))
        conv_outputs = conv_outputs[0].numpy()

        # 5. Compute CAM
        cam = np.zeros(conv_outputs.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_outputs[:, :, i]
        
        # 6. Normalize CAM
        cam = np.maximum(cam, 0)
        cam = cam / (np.max(cam) + 1e-10)
        
        # 7. Prepare images
        cam_uint8 = np.uint8(255 * cam)
        original_img = np.uint8(img_array[0] * 255)
        
        # 8. Resize and apply colormap
        cam_resized = cv2.resize(cam_uint8, (original_img.shape[1], original_img.shape[0]))
        heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
        
        # 9. Convert to RGB and superimpose
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_rgb, 0.4, 0)
        
        # 10. Encode to base64
        _, buf = cv2.imencode('.jpg', superimposed_img)
        return base64.b64encode(buf).decode('utf-8')
    
    except Exception as e:
        print(f"GradCAM Error: {str(e)}")
        # Fallback: Return original image if GradCAM fails
        original_img = np.uint8(img_array[0] * 255)
        _, buf = cv2.imencode('.jpg', original_img)
        return base64.b64encode(buf).decode('utf-8')