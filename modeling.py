import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("dataset.csv")

# =========================
# 2. LABELING
# =========================
def label_dehidrasi(pH):
    if pH < 5.0:
        return "Dehidrasi_Berat"
    elif 5.0 <= pH < 5.5:
        return "Dehidrasi_Ringan"
    else:
        return "Normal"

df['label_dehidrasi'] = df['pH'].apply(label_dehidrasi)

df['label_diabetes'] = df['Label'].apply(
    lambda x: 1 if 'Diabetes' in str(x) else 0
)

df['label_ginjal'] = df['Label'].apply(
    lambda x: 1 if 'Ginjal' in str(x) else 0
)

# =========================
# 3. FITUR
# =========================
X = df[['pH', 'R', 'G', 'B']]

y_knn = df['label_dehidrasi']
y_nb = df['label_diabetes']
y_svm = df['label_ginjal']

# =========================
# 4. SPLIT
# =========================
X_train, X_test, y_knn_train, y_knn_test, y_nb_train, y_nb_test, y_svm_train, y_svm_test = train_test_split(
    X, y_knn, y_nb, y_svm,
    test_size=0.2,
    random_state=42
)

# =========================
# 5. SCALING
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# 6. TRAIN MODEL
# =========================
knn_model = KNeighborsClassifier(n_neighbors=5)
nb_model = GaussianNB()
svm_model = SVC(kernel='rbf')

knn_model.fit(X_train_scaled, y_knn_train)
nb_model.fit(X_train_scaled, y_nb_train)
svm_model.fit(X_train_scaled, y_svm_train)

# =========================
# 7. PREDIKSI
# =========================
pred_knn = knn_model.predict(X_test_scaled)
pred_nb = nb_model.predict(X_test_scaled)
pred_svm = svm_model.predict(X_test_scaled)

# =========================
# 8. AKURASI
# =========================
print("\n=== AKURASI ===")
acc_knn = accuracy_score(y_knn_test, pred_knn)
acc_nb = accuracy_score(y_nb_test, pred_nb)
acc_svm = accuracy_score(y_svm_test, pred_svm)

print("KNN (Dehidrasi):", acc_knn)
print("NB (Diabetes):", acc_nb)
print("SVM (Ginjal):", acc_svm)

# =========================
# 9. EVALUASI DETAIL KNN
# =========================
print("\n=== HASIL KNN (3 KELAS) ===")

print(f"Accuracy: {acc_knn*100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_knn_test,
    pred_knn,
    target_names=["Normal", "Dehidrasi Ringan", "Dehidrasi Berat"]
))

# =========================
# 10. CONFUSION MATRIX KNN
# =========================
cm_knn = confusion_matrix(y_knn_test, pred_knn)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm_knn,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Normal", "Dehidrasi Ringan", "Dehidrasi Berat"],
    yticklabels=["Normal", "Dehidrasi Ringan", "Dehidrasi Berat"]
)

plt.title('Confusion Matrix - KNN (3 Kelas)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# =========================
# 11. SAVE MODEL
# =========================
model_bundle = {
    "knn_model": knn_model,
    "nb_model": nb_model,
    "svm_model": svm_model,
    "scaler": scaler
}

with open("model_urine.sav", "wb") as f:
    pickle.dump(model_bundle, f)

print("\nModel berhasil disimpan!")
