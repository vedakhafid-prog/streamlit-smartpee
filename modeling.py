import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("dataset.csv")

# =========================
# 2. BUAT LABEL (JIKA PERLU)
# =========================
if 'Label' in df.columns:
    df['dehidrasi'] = df['Label'].apply(lambda x: 1 if 'Dehidrasi' in str(x) else 0)
    df['diabetes']  = df['Label'].apply(lambda x: 1 if 'Diabetes' in str(x) else 0)
    df['ginjal']    = df['Label'].apply(lambda x: 1 if 'Ginjal' in str(x) else 0)

# =========================
# 3. FITUR
# =========================
X = df[['pH', 'R', 'G', 'B']]

y_deh = df['dehidrasi']
y_dm = df['diabetes']
y_ginjal = df['ginjal']

# =========================
# 4. SPLIT
# =========================
X_train, X_test, y_deh_train, y_deh_test, y_dm_train, y_dm_test, y_ginjal_train, y_ginjal_test = train_test_split(
    X, y_deh, y_dm, y_ginjal,
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
# 6. RULE GINJAL
# =========================
def rule_ginjal(pH):
    return 1 if (pH < 5.0 or pH > 8.0) else 0

# =========================
# 7. TRAIN MODEL
# =========================
knn_model = KNeighborsClassifier(n_neighbors=5)
nb_model = GaussianNB()
svm_model = SVC(kernel='rbf')

knn_model.fit(X_train_scaled, y_deh_train)
nb_model.fit(X_train_scaled, y_dm_train)
svm_model.fit(X_train_scaled, y_ginjal_train)

# =========================
# 8. EVALUASI
# =========================
pred_knn = knn_model.predict(X_test_scaled)
pred_nb = nb_model.predict(X_test_scaled)
pred_svm = svm_model.predict(X_test_scaled)

print("=== AKURASI ===")
print("KNN:", accuracy_score(y_deh_test, pred_knn))
print("NB:", accuracy_score(y_dm_test, pred_nb))
print("SVM:", accuracy_score(y_ginjal_test, pred_svm))

# =========================
# 9. SAVE MODEL (1 FILE)
# =========================
model_bundle = {
    "knn_model": knn_model,
    "nb_model": nb_model,
    "svm_model": svm_model,
    "scaler": scaler
}

with open("model_urine.sav", "wb") as f:
    pickle.dump(model_bundle, f)

print("Model tersimpan sebagai model_urine.sav")