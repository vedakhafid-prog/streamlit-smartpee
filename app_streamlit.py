import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model_svm_urine.sav','rb'))

st.title('Klasifikasi Kualitas Urine (SVM)')

ph = st.number_input('pH', 0.0, 14.0, 6.0)
R = st.number_input('R', 0.0, 255.0, 100.0)
G = st.number_input('G', 0.0, 255.0, 100.0)
B = st.number_input('B', 0.0, 255.0, 100.0)

if st.button('Prediksi'):
    data = np.array([[ph,R,G,B]])
    pred = model.predict(data)
    st.write('Hasil:', pred[0])
