import streamlit as st
import joblib
import numpy as np

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Deteksi Kesehatan Urine", layout="centered")

# 2. Fungsi untuk Memuat Model (Cached agar lebih cepat)
@st.cache_resource
def load_model():
    # Pastikan nama file sesuai dengan yang Anda simpan sebelumnya
    return joblib.load('model_urine_lengkap.sav')

def main():
    st.title("🧪 Sistem Deteksi Kesehatan Urine")
    st.write("Masukkan nilai parameter di bawah ini untuk memprediksi status kesehatan ginjal.")

    try:
        # Memuat komponen dari file .sav tunggal
        data_load = load_model()
        model = data_load['model']
        scaler = data_load['scaler']
        le = data_load['label_encoder']

        # 3. Form Input User
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                ph = st.number_input("Nilai pH", min_value=0.0, max_value=14.0, value=6.0, step=0.1)
                r = st.number_input("Nilai Red (R)", min_value=0, max_value=255, value=255)
                
            with col2:
                g = st.number_input("Nilai Green (G)", min_value=0, max_value=255, value=255)
                b = st.number_input("Nilai Blue (B)", min_value=0, max_value=255, value=0)

            submit = st.form_submit_button("Prediksi")

        # 4. Logika Prediksi
        if submit:
            # Data input dalam bentuk array 2D
            input_data = np.array([[ph, r, g, b]])
            
            # Wajib di-scale menggunakan scaler yang sama saat training
            input_scaled = scaler.transform(input_data)
            
            # Melakukan prediksi
            prediction = model.predict(input_scaled)
            result_label = le.inverse_transform(prediction)
            
            # 5. Menampilkan Hasil
            st.divider()
            st.subheader("Hasil Analisis:")
            
            if result_label == "Normal":
                st.success(f"Status: **{result_label}**")
                st.balloons()
            else:
                st.error(f"Status: **{result_label}**")
                st.warning("Perhatian: Hasil menunjukkan indikasi gangguan fungsi ginjal.")

    except FileNotFoundError:
        st.error("Error: File 'model_urine_lengkap.sav' tidak ditemukan di folder yang sama.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()