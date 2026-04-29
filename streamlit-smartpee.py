import streamlit as st
import numpy as np
import pickle

# ========================
# LOAD MODEL (AMAN)
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

pH = st.slider("pH", 4.0, 9.0, 6.0, step=0.1)
R = st.slider("Red (R)", 0, 255, 150)
G = st.slider("Green (G)", 0, 255, 150)
B = st.slider("Blue (B)", 0, 255, 150)

# ========================
# PREDIKSI
# ========================
if st.button("🔍 Prediksi"):
    try:
        data = np.array([[pH, R, G, B]])
        data_scaled = scaler.transform(data)

        # Prediksi
        deh = knn.predict(data_scaled)[0]
        dm = nb.predict(data_scaled)[0]
        ginjal_ml = svm.predict(data_scaled)[0]

        # Rule + ML
        ginjal_rule = rule_ginjal(pH)
        ginjal_final = 1 if (ginjal_ml or ginjal_rule) else 0

        # ========================
        # OUTPUT
        # ========================
        st.subheader("📊 Hasil Analisis")

        st.success(f"Dehidrasi: {'Ya' if deh else 'Tidak'}")
        st.success(f"Diabetes: {'Ya' if dm else 'Tidak'}")

        if ginjal_final:
            st.error("Ginjal: Berisiko ⚠️")
        else:
            st.success("Ginjal: Normal ✅")

    except Exception as e:
        st.error(f"Terjadi error: {e}")