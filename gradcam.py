import io
import base64
import cv2
import numpy as np
import tensorflow as tf

from PIL import Image


def make_gradcam_heatmap(
    img_array,
    model,
    last_conv_layer_name,
    pred_index
):
    # Model khusus untuk Grad-CAM
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )
    
    # Hitung gradient
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if pred_index == 0:
            class_channel = 1.0 - predictions[:, 0]
        else:
            class_channel = predictions[:, 0]

    # Gradient terhadap feature map
    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    # Global average pooling
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    # Ambil feature map pertama
    conv_outputs = conv_outputs[0]

    # Weighted feature map
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalisasi
    max_val = tf.math.reduce_max(heatmap)

    if max_val == 0:
        return np.zeros_like(heatmap.numpy())

    heatmap = heatmap / max_val

    return heatmap.numpy()


def overlay_heatmap(
    image_bytes,
    heatmap,
    alpha=0.4,
    image_size=(160, 160)
):
    # Load original image
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(image_size)

    original_img = np.array(img)

    # Convert heatmap
    heatmap = np.uint8(255 * heatmap)

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        image_size
    )

    # Apply color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Overlay
    superimposed_img = cv2.addWeighted(
        original_img,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return superimposed_img


def image_to_base64(image_array):
    # Convert numpy array → PNG
    _, buffer = cv2.imencode(".png", image_array)

    # Convert → base64
    image_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return image_base64