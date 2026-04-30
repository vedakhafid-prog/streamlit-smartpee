import streamlit as st
import numpy as np
import pickle

# ========================
# LOAD MODEL
# ========================
@st.cache_resource
def load_model():
    with open("model_urine.sav", "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_model()

knn = bundle["knn_model"]
nb = bundle["nb_model"]
svm = bundle["svm_model"]
scaler = bundle["scaler"]

# ========================
# RULE BASED
# ========================
def rule_ginjal(pH):
    return 1 if (pH < 5 or pH > 8) else 0

# ========================
# UI
# ========================
st.title("💧 Smart Pee Detection")
st.write("Masukkan nilai pH dan warna urine (RGB)")

pH = st.number_input("pH", min_value=4.0, max_value=9.0, value=6.0, step=0.1)
R = st.number_input("Red (R)", min_value=0, max_value=255, value=150)
G = st.number_input("Green (G)", min_value=0, max_value=255, value=150)
B = st.number_input("Blue (B)", min_value=0, max_value=255, value=150)

# ========================
# PREDIKSI
# ========================
if st.button("🔍 Prediksi"):
    try:
        data = np.array([[pH, R, G, B]])
        data_scaled = scaler.transform(data)

        # ========================
        # PREDIKSI MODEL
        # ========================
        hasil_knn = knn.predict(data_scaled)[0]   # STRING
        hasil_nb = nb.predict(data_scaled)[0]     # 0 / 1
        hasil_svm = svm.predict(data_scaled)[0]   # 0 / 1

        # ========================
        # RULE + ML (GINJAL)
        # ========================
        ginjal_rule = rule_ginjal(pH)
        ginjal_final = 1 if (hasil_svm or ginjal_rule) else 0

        # ========================
        # OUTPUT
        # ========================
        st.subheader("📊 Hasil Analisis")

        # --- KNN (Dehidrasi 3 kelas) ---
        if hasil_knn == "Normal":
            st.success("Dehidrasi: Normal ✅")
        elif hasil_knn == "Dehidrasi_Ringan":
            st.warning("Dehidrasi: Ringan ⚠️")
        elif hasil_knn == "Dehidrasi_Berat":
            st.error("Dehidrasi: Berat 🚨")
        else:
            st.info(f"Dehidrasi: {hasil_knn}")

        # --- NB (Diabetes) ---
        if hasil_nb == 1:
            st.error("Diabetes: Terindikasi ⚠️")
        else:
            st.success("Diabetes: Normal ✅")

        # --- SVM + Rule (Ginjal) ---
        if ginjal_final == 1:
            st.error("Ginjal: Terindikasi Gangguan ⚠️")
        else:
            st.success("Ginjal: Normal ✅")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
