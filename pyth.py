from fastapi import FastAPI, UploadFile, File, Form
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from io import BytesIO
import torch
import mysql.connector
from mysql.connector import Error
import json
import os
import numpy as np

# ----------------- MODEL LOAD -----------------

model_id = "codewithdark/vit-chest-xray"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id)

app = FastAPI()

# ----------------- DATABASE CONFIG -----------------

db_config = {
    "host": "localhost",
    "user": "root",
    "password": "MyNewPass123!",   # your MySQL password
    "database": "xray_db"
}

def get_db_connection():
    try:
        connection = mysql.connector.connect(**db_config)
        return connection
    except Error as e:
        print(f"Database connection error: {e}")
        return None

def save_prediction(patient_id, patient_name, age, symptoms, lang, label, confidence, risk, refer):
    connection = get_db_connection()
    if connection:
        try:
            cursor = connection.cursor()
            sql = """
            INSERT INTO patient_history
            (patient_id, patient_name, age, symptoms, lang, label, confidence, risk, refer)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (patient_id, patient_name, age, symptoms, lang, label, confidence, risk, int(refer))
            cursor.execute(sql, values)
            connection.commit()
        except Error as e:
            print(f"Database insert error: {e}")
        finally:
            cursor.close()
            connection.close()

# ----------------- JSON HISTORY -----------------

HISTORY_FILE = "history.json"

def save_history_json(patient_id, patient_name, age, symptoms, lang, result):
    record = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": age,
        "symptoms": symptoms,
        "lang": lang,
        "result": result
    }

    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(record)

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ----------------- ANALYSIS LOGIC -----------------

def analyze_xray(image: Image.Image, age: int, symptoms: str, lang: str = "en"):
    # Blur check
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float32)
    blur_score = float(arr.var())

    blur_warning = ""
    if blur_score < 50.0:
        blur_warning = "Image may be blurry/low quality. Please retake X-ray."

    # Model inference
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=-1)

    top2 = torch.topk(probs, k=2)
    top_indices = top2.indices.tolist()
    top_scores = top2.values.tolist()

    id2label = model.config.id2label

    label_map = {
        "LABEL_0": "Normal",
        "LABEL_1": "Pneumonia",
        "LABEL_2": "Tuberculosis",
        "LABEL_3": "Cardiomegaly",
        "LABEL_4": "Other abnormality"
    }

    def decode(idx, score):
        label_id = id2label[idx]
        human = label_map.get(label_id, label_id)
        return {
            "index": int(idx),
            "id_label": label_id,
            "label": human,
            "score": float(score)
        }

    top_pred_1 = decode(top_indices[0], top_scores[0])
    top_pred_2 = decode(top_indices[1], top_scores[1]) if len(top_indices) > 1 else None

    label = top_pred_1["label"]
    score = top_pred_1["score"]

    # Risk logic
    risk = "LOW"
    refer = False
    symptoms_lower = symptoms.lower()

    if age >= 60 or "cough" in symptoms_lower or "fever" in symptoms_lower:
        risk = "HIGH"
        refer = True

    # Messages
    msg_en = {
        "Normal": "No serious abnormality detected.",
        "Pneumonia": "Pneumonia suspected.",
        "Tuberculosis": "Tuberculosis suspected.",
        "Cardiomegaly": "Heart appears enlarged.",
        "Other abnormality": "Some abnormal changes detected."
    }

    msg_te = {
        "Normal": "ఎటువంటి గంభీర సమస్య కనిపించలేదు.",
        "Pneumonia": "న్యుమోనియా అవకాశం ఉంది.",
        "Tuberculosis": "టి.బి. అనుమానం ఉంది.",
        "Cardiomegaly": "గుండె పెద్దగా కనిపిస్తోంది.",
        "Other abnormality": "కొన్ని అసాధారణ మార్పులు ఉన్నాయి."
    }

    msg_hi = {
        "Normal": "कोई गंभीर समस्या नहीं दिखी।",
        "Pneumonia": "निमोनिया की संभावना है।",
        "Tuberculosis": "टीबी की संभावना है।",
        "Cardiomegaly": "दिल बड़ा दिख रहा है।",
        "Other abnormality": "कुछ असामान्य बदलाव दिखे हैं।"
    }

    msg_kn = {
        "Normal": "ಯಾವುದೇ ಗಂಭೀರ ಅಸಾಮಾನ್ಯತೆ ಕಾಣಿಸಲಿಲ್ಲ.",
        "Pneumonia": "ನಿಮೋನಿಯಾ ಇರುವ ಸಾಧ್ಯತೆ ಇದೆ.",
        "Tuberculosis": "ಕ್ಷಯರೋಗ (TB) ಇರುವ ಅನುಮಾನ ಇದೆ.",
        "Cardiomegaly": "ಹೃದಯವು ದೊಡ್ಡದಾಗಿ ಕಾಣುತ್ತಿದೆ.",
        "Other abnormality": "ಕೆಲವು ಅಸಾಮಾನ್ಯ ಬದಲಾವಣೆಗಳು ಕಂಡುಬಂದಿವೆ."
    }

    lang_norm = lang.lower()
    if lang_norm == "te":
        message = msg_te.get(label, "")
    elif lang_norm == "hi":
        message = msg_hi.get(label, "")
    elif lang_norm == "kn":
        message = msg_kn.get(label, "")
    else:
        message = msg_en.get(label, "")

    if blur_warning:
        message = message + " " + blur_warning

    return {
        "label": label,
        "confidence": score,
        "risk": risk,
        "refer": refer,
        "language": lang_norm,
        "message": message,
        "blur_score": blur_score,
        "blur_warning": blur_warning,
        "top2": [top_pred_1] + ([top_pred_2] if top_pred_2 else [])
    }

# ----------------- ENDPOINTS -----------------

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    age: int = Form(...),
    symptoms: str = Form(...),
    lang: str = Form("en"),
    patient_id: str = Form("demo_id"),
    patient_name: str = Form("demo_user")
):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).convert("RGB")

    result = analyze_xray(image, age, symptoms, lang)

    # Save to DB
    save_prediction(
        patient_id=patient_id,
        patient_name=patient_name,
        age=age,
        symptoms=symptoms,
        lang=lang,
        label=result["label"],
        confidence=result["confidence"],
        risk=result["risk"],
        refer=result["refer"]
    )

    # Save to JSON file
    save_history_json(
        patient_id=patient_id,
        patient_name=patient_name,
        age=age,
        symptoms=symptoms,
        lang=lang,
        result=result
    )

    return {
        "message": "Analysis complete",
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": age,
        "symptoms": symptoms,
        "result": result
    }

@app.post("/login")
async def login(
    username: str = Form(...),
    password: str = Form(...),
    patient_name: str = Form(...)
):
    # For now, accept any username/password
    return {
        "success": True,
        "role": "user",
        "username": username,
        "patient_name": patient_name,
        "message": "Login successful"
    }
