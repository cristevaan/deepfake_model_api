import os
import io
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from tensorflow.keras.models import load_model
from groq import Groq

from gradcam import (
    make_gradcam_heatmap,
    overlay_heatmap,
    image_to_base64
)


# Config
app = FastAPI()
MODEL_PATH = "model/model.keras"
LABEL_PATH = "model/label.txt"
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# Resource Validation
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"{MODEL_PATH} tidak ditemukan")

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"{LABEL_PATH} tidak ditemukan")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY belum diatur")


# Load Resource
model = load_model(MODEL_PATH)

with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

client = Groq(api_key=GROQ_API_KEY)


# Image Preprocessing Function
def preprocess_image(
    image_bytes,
    image_size=(160, 160),
    color_mode="rgb"
    ):

    img = Image.open(io.BytesIO(image_bytes))
    
    if color_mode == "rgb":
        img = img.convert("RGB")
    else:
        img = img.convert("L")
    
    img = img.resize(image_size)
    img_array = np.array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# Call Generative AI
def call_genai(result_data):
    try:
        messages = [
            {
                "role": "system",
                "content": """
Anda adalah AI penjelas hasil deteksi deepfake foto atau gambar bukan video.

Aturan:
- Jangan gunakan markdown.
- Kembalikan teks biasa.
- Jangan mengubah hasil prediksi.
- Jangan membuat angka baru.
- Gunakan data yang diberikan saja.
- Maksimal 4 kalimat.
- Gunakan bahasa Indonesia.
- Jangan menyatakan hasil sebagai kepastian mutlak.
- Berikan saran singkat untuk verifikasi tambahan.
"""
            },
            {
                "role": "user",
                "content": str(result_data)
            }
        ]

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages
        )

        return response.choices[0].message.content

    except Exception as e:
        print("GenAI Error:", e)
        
        return (
            "Penjelasan otomatis tidak tersedia saat ini."
        )


# Prediction Endpoint
@app.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    threshold: float = 0.5
):
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail="File harus berupa gambar"
            )

        # Read image
        image_bytes = await file.read()

        # Preprocess
        img_array = preprocess_image(
            image_bytes=image_bytes
        )

        # Predict
        prediction = model.predict(
            img_array,
            verbose=0
        )

        # Validate output
        if prediction.shape != (1, 1):
            
            raise HTTPException(
                status_code=500,
                detail="Format output model tidak sesuai"
            )

        # Interpret results
        score = float(prediction[0][0])
        probabilities = {
            labels[0]: round((1 - score) * 100, 2),
            labels[1]: round(score * 100, 2)
        }
        pred_index = (1 if score > threshold else 0)
        pred_label = labels[pred_index]
        confidence = round(probabilities[pred_label], 2)
        
        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(
            img_array=img_array,
            model=model,
            last_conv_layer_name="activation_8",
            pred_index=pred_index
        )

        # Overlay heatmap ke gambar asli
        gradcam_image = overlay_heatmap(
            image_bytes=image_bytes,
            heatmap=heatmap
        )

        # Convert ke base64
        gradcam_base64 = image_to_base64(
            gradcam_image
        )

        # Data for GenAI
        result_data = {
            "prediction": pred_label,
            "confidence": confidence
        }

        # Get explanation from GenAI
        explanation = call_genai(
            result_data
        )

        # Final response
        return {
            "filename": file.filename,
            "prediction": pred_label,
            "confidence": confidence,
            "probabilities": probabilities,
            "raw_score": score,
            "explanation": explanation,
            "gradcam": gradcam_base64
        }

    except HTTPException:
        raise

    except Exception as e:
        print("Prediction Error:", e)

        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat prediksi"
        )