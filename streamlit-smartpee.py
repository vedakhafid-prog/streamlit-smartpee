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

acc_knn = bundle["acc_knn"]
acc_nb = bundle["acc_nb"]
acc_svm = bundle["acc_svm"]

st.subheader("📈 Akurasi Model")
st.write(f"KNN: {acc_knn*100:.2f}%")
st.write(f"Naive Bayes: {acc_nb*100:.2f}%")
st.write(f"SVM: {acc_svm*100:.2f}%")
