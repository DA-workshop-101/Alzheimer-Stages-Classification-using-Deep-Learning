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


def generate_gradcam(image_tensor, model, layer_name="block5_conv3"):
    grad_model = tf.keras.Model(
        inputs=[model.input],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_tensor)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Resize heatmap using PIL
    heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize((255, 255), resample=Image.BILINEAR)
    heatmap = np.array(heatmap)

    # Apply colormap using matplotlib
    import matplotlib.cm as cm
    colormap = plt.colormaps['jet']
    heatmap_colored = colormap(heatmap / 255.0)  # returns RGBA
    heatmap_colored = np.uint8(255 * heatmap_colored[:, :, :3])  # Drop alpha

    return heatmap_colored

# def generate_gradcam(img_array, class_index):
#     """
#     Simplified GradCAM implementation with layer selection based on model's summary.
#     """
#     try:
#         # Get the third-to-last convolutional layer using model.get_layer
#         last_conv_layer = model.get_layer(index=-4)

#         # Create the gradient model
#         grad_model = tf.keras.models.Model(
#             [model.inputs], [last_conv_layer.output, model.output]
#         )

#         # Calculate gradients
#         with tf.GradientTape() as tape:
#             conv_outputs, predictions = grad_model(img_array)
#             loss = predictions[:, class_index]

#         # Get gradients and weights
#         grads = tape.gradient(loss, conv_outputs)[0]
#         conv_outputs = conv_outputs[0]

#         # Compute weights by averaging the gradients across spatial dimensions
#         if len(grads.shape) == 3:  # (H, W, Channels)
#             weights = tf.reduce_mean(grads, axis=(0, 1))
#         elif len(grads.shape) == 1:  # Fallback if the gradient is already 1D
#             weights = grads
#         else:
#             raise ValueError(f"Unexpected grads shape: {grads.shape}")

#         # Compute the Class Activation Map (CAM)
#         cam = np.dot(conv_outputs, weights.numpy())

#         # Normalize and resize the CAM
#         cam = np.maximum(cam, 0)  # ReLU
#         cam = cam / cam.max()  # Normalize
#         cam = cv2.resize(cam, (224, 224))

#         # Apply a colormap
#         heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

#         # Convert the image to uint8 for visualization
#         img = np.uint8(img_array[0] * 255)
#         superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

#         # Convert to a PIL image and encode to base64
#         pil_img = Image.fromarray(superimposed_img)
#         buf = io.BytesIO()
#         pil_img.save(buf, format='PNG')
#         encoded_img = base64.b64encode(buf.getvalue()).decode('utf-8')

#         return encoded_img
    
#     except Exception as e:
#         print(f"Error during GradCAM generation: {e}")
#         return None
