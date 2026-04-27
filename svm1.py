import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Memuat Dataset
df = pd.read_csv('urine_labeled.csv')

# 2. Pra-pemrosesan Data
# Mengubah label teks (Status) menjadi angka
le = LabelEncoder()
df['Status_Encoded'] = le.fit_transform(df['Status'])

# Memisahkan Fitur (X) dan Target (y)
X = df[['pH', 'R', 'G', 'B']]
y = df['Status_Encoded']

# Membagi data menjadi Training Set (80%) dan Test Set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi Fitur (Scaling) agar SVM bekerja optimal
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Membangun dan Melatih Model SVM
# Menggunakan kernel 'linear' (bisa diganti 'rbf' atau 'poly' jika data lebih kompleks)
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 4. Prediksi dan Evaluasi
y_pred = svm_model.predict(X_test_scaled)

# Menampilkan Hasil Evaluasi
print(f"Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 5. Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix - SVM Model')
plt.show()