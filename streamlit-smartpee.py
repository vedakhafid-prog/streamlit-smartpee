import streamlit as st
import numpy as np
import pickle

# ========================
# CONFIG
# ========================
MODEL_PATH = "model_urine.sav"

# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)

bundle = load_model(MODEL_PATH)

knn_model = bundle["knn_model"]
nb_model = bundle["nb_model"]
svm_model = bundle["svm_model"]
scaler = bundle["scaler"]

# ========================
# RULE ENGINE
# ========================
def rule_ginjal(pH: float) -> int:
    return 1 if (pH < 5 or pH > 8) else 0

# ========================
# PREDICTION ENGINE
# ========================
def predict(data):
    data_scaled = scaler.transform(data)

    hasil_knn = knn_model.predict(data_scaled)[0]
    hasil_nb = nb_model.predict(data_scaled)[0]
    hasil_svm = svm_model.predict(data_scaled)[0]

    ginjal_rule = rule_ginjal(data[0][0])
    ginjal_final = 1 if (hasil_svm or ginjal_rule) else 0

    return {
        "dehidrasi": hasil_knn,
        "diabetes": hasil_nb,
        "ginjal": ginjal_final
    }

# ========================
# FORMAT OUTPUT
# ========================
def format_dehidrasi(label):
    # HANDLE MODEL LAMA (0/1)
    if isinstance(label, (int, np.integer)):
        if label == 0:
            return "Normal", "success"
        elif label == 1:
            return "Dehidrasi", "warning"

    # HANDLE MODEL BARU (MULTI-CLASS)
    mapping = {
        "Normal": ("Normal", "success"),
        "Dehidrasi_Ringan": ("Dehidrasi Ringan", "warning"),
        "Dehidrasi_Berat": ("Dehidrasi Berat", "error"),
    }

    return mapping.get(label, (str(label), "info"))

def format_binary(value, positive_text):
    if value == 1:
        return positive_text, "error"
    return "Normal", "success"

# ========================
# UI
# ========================
st.set_page_config(page_title="Smart Pee Detection", layout="centered")

st.title("💧 Smart Pee Detection")
st.caption("Deteksi kondisi urine berbasis pH dan warna (RGB)")

col1, col2 = st.columns(2)

with col1:
    pH = st.number_input("pH", 4.0, 9.0, 6.0, step=0.1)

with col2:
    R = st.number_input("R", 0, 255, 150)
    G = st.number_input("G", 0, 255, 150)
    B = st.number_input("B", 0, 255, 150)

# ========================
# RUN
# ========================
if st.button("🔍 Prediksi"):
    try:
        data = np.array([[pH, R, G, B]])

        result = predict(data)

        st.subheader("📊 Hasil Analisis")

        # DEBUG (biar kamu tahu model output apa)
        st.caption(f"DEBUG KNN: {result['dehidrasi']} ({type(result['dehidrasi'])})")

        # --- Dehidrasi ---
        text, level = format_dehidrasi(result["dehidrasi"])
        getattr(st, level)(f"Dehidrasi: {text}")

        # --- Diabetes ---
        text, level = format_binary(result["diabetes"], "Terindikasi Diabetes Mellitus")
        getattr(st, level)(f"Diabetes: {text}")

        # --- Ginjal ---
        text, level = format_binary(result["ginjal"], "Terindikasi Gangguan Fungsi Ginjal")
        getattr(st, level)(f"Ginjal: {text}")

    except Exception as e:
        st.error(f"Error: {e}")
