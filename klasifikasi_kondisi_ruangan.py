"""
Proyek UTS: Klasifikasi Kondisi Ruangan
Fitur: suhu (°C), kelembapan (%RH), cahaya (lux)
Model: Decision Tree
Catatan: Dataset merupakan simulasi; hasil bukan validasi kondisi ruangan nyata.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# 1. Membaca dataset
DATA_PATH = Path("dataset_simulasi_kondisi_ruangan.csv")
df = pd.read_csv(DATA_PATH)

fitur = ["suhu", "kelembapan", "cahaya"]
target = "kondisi_ruangan"

print("=" * 65)
print("PROYEK UTS - KLASIFIKASI KONDISI RUANGAN")
print("=" * 65)
print("\nUkuran dataset:", df.shape)
print("\nLima data pertama:")
print(df.head())
print("\nInformasi dataset:")
print(df.info())
print("\nJumlah missing values sebelum preprocessing:")
print(df[fitur].isna().sum())
print("\nDistribusi kelas:")
print(df[target].value_counts())

# 2. Validasi rentang fisik sederhana (hanya menandai, tidak otomatis menghapus)
rentang_fisik = {
    "suhu": (0, 50),
    "kelembapan": (0, 100),
    "cahaya": (0, 2000)
}
print("\nPemeriksaan nilai di luar rentang fisik yang diasumsikan:")
for kolom, (bawah, atas) in rentang_fisik.items():
    mask = df[kolom].notna() & ~df[kolom].between(bawah, atas)
    print(f"{kolom}: {int(mask.sum())} nilai")

# 3. Visualisasi eksplorasi data
plt.figure(figsize=(7, 4))
kelas_urut = ["Nyaman", "Kurang Nyaman", "Tidak Nyaman"]
jumlah_kelas = df[target].value_counts().reindex(kelas_urut, fill_value=0)
plt.bar(jumlah_kelas.index, jumlah_kelas.values)
plt.title("Distribusi Kondisi Ruangan")
plt.xlabel("Kondisi Ruangan")
plt.ylabel("Jumlah Data")
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
korelasi = df[fitur].corr()
plt.imshow(korelasi, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Koefisien korelasi")
plt.xticks(range(len(fitur)), fitur, rotation=15)
plt.yticks(range(len(fitur)), fitur)
for i in range(len(fitur)):
    for j in range(len(fitur)):
        plt.text(j, i, f"{korelasi.iloc[i, j]:.2f}", ha="center", va="center")
plt.title("Korelasi Antarfitur Sensor")
plt.tight_layout()
plt.show()

# 4. Memisahkan fitur dan label
X = df[fitur]
y = df[target]

# 5. Train-test split sebelum fitting preprocessing untuk mencegah data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing :", len(X_test))

# 6. Pipeline preprocessing + model.
# Missing values diisi median berdasarkan data training.
# Decision Tree tidak membutuhkan normalisasi fitur.
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", DecisionTreeClassifier(
        criterion="gini",
        max_depth=5,
        min_samples_leaf=4,
        random_state=42
    ))
])

# 7. Training
model.fit(X_train, y_train)

# 8. Prediksi dan evaluasi pada data testing
y_pred = model.predict(X_test)

print("\nAkurasi:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

labels = ["Nyaman", "Kurang Nyaman", "Tidak Nyaman"]
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Decision Tree")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# 9. Visualisasi pohon keputusan
plt.figure(figsize=(18, 9))
plot_tree(
    model.named_steps["classifier"],
    feature_names=fitur,
    class_names=model.named_steps["classifier"].classes_,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Pohon Keputusan Klasifikasi Kondisi Ruangan")
plt.tight_layout()
plt.show()

# 10. Feature importance
importance = pd.Series(
    model.named_steps["classifier"].feature_importances_,
    index=fitur
).sort_values(ascending=False)
print("\nFeature importance:")
print(importance)

importance.sort_values().plot(kind="barh", figsize=(7, 4))
plt.title("Feature Importance Decision Tree")
plt.xlabel("Tingkat Kepentingan")
plt.tight_layout()
plt.show()

# 11. Contoh prediksi satu kondisi baru
data_baru = pd.DataFrame([{
    "suhu": 26.0,
    "kelembapan": 55.0,
    "cahaya": 450.0
}])
prediksi = model.predict(data_baru)[0]
print("\nContoh data baru:")
print(data_baru)
print("Prediksi kondisi ruangan:", prediksi)

# 12. Simpan model agar dapat digunakan kembali
joblib.dump(model, "model_kondisi_ruangan.joblib")
print("\nModel tersimpan sebagai model_kondisi_ruangan.joblib")

# Catatan interpretasi:
# Label dataset dibentuk menggunakan aturan ambang yang ditentukan pada proyek.
# Akurasi tinggi dapat terjadi karena model mempelajari aturan label simulasi.
# Uji lanjutan dengan data sensor nyata diperlukan untuk menilai penerapan sebenarnya.
