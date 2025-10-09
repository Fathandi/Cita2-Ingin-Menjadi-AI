"""
Fashion MNIST Custom Image Predictor
=====================================
Script untuk melakukan prediksi pada gambar custom menggunakan model yang sudah dilatih.
Fitur:
1. Klasifikasi gambar ke 10 kategori fashion
2. Organisasi hasil ke folder berdasarkan label
3. Visualisasi distribusi prediksi dengan bar chart
4. Confidence score untuk setiap prediksi

Dibuat oleh: Fathih Apriandi
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt

# ============================================================================
# KONFIGURASI LABEL FASHION MNIST
# ============================================================================

# Label resmi dataset Fashion MNIST (sesuai urutan index 0-9)
# Setiap index mewakili satu kategori fashion item
LABELS = [
    "T-shirt/top",    # Index 0: T-shirt/top
    "Trouser",        # Index 1: Trouser
    "Pullover",       # Index 2: Pullover
    "Dress",          # Index 3: Dress
    "Coat",           # Index 4: Coat
    "Sandal",         # Index 5: Sandal
    "Shirt",          # Index 6: Shirt
    "Sneaker",        # Index 7: Sneaker
    "Bag",            # Index 8: Bag
    "Ankle boot"      # Index 9: Ankle boot
]

# ============================================================================
# LOAD MODEL YANG SUDAH DILATIH
# ============================================================================

# Memuat model Fashion MNIST yang sudah dilatih
# Model berisi arsitektur neural network dan weights hasil training
# Format .keras: format modern TensorFlow untuk model saving/loading
print("🔄 Memuat model yang sudah dilatih...")
model = tf.keras.models.load_model('fashion_mnist_model.keras')
print("✅ Model berhasil dimuat!")

# ============================================================================
# KONFIGURASI FOLDER INPUT & OUTPUT
# ============================================================================

# Folder input: berisi gambar yang akan diprediksi
# Expected: gambar PNG/JPG/JPEG dari download_image.py
INPUT_FOLDER = 'test-image'

# Folder output: untuk menyimpan hasil klasifikasi
# Struktur: result/label/gambar.png
OUTPUT_FOLDER = 'result'

# ============================================================================
# SETUP FOLDER OUTPUT
# ============================================================================

# Hapus folder result jika sudah ada (clean start)
# shutil.rmtree: menghapus recursive seluruh isi folder
# Tujuan: menghindari duplikasi file dari run sebelumnya
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
    print(f"🗑️  Folder '{OUTPUT_FOLDER}' lama dihapus")

# Buat folder result utama
os.makedirs(OUTPUT_FOLDER)
print(f"📁 Folder '{OUTPUT_FOLDER}' dibuat")

# Buat subfolder untuk setiap label fashion
# Setiap kategori mendapat folder terpisah untuk organisasi hasil
for label in LABELS:
    label_path = os.path.join(OUTPUT_FOLDER, label)
    os.makedirs(label_path, exist_ok=True)
    print(f"   📂 Subfolder '{label}' dibuat")

print("✅ Struktur folder output siap!")

# ============================================================================
# INISIALISASI STATISTIK PREDIKSI
# ============================================================================

# Dictionary untuk melacak jumlah prediksi per label
# Format: {label: count} dengan initial value 0 untuk semua label
# Digunakan untuk analisis distribusi hasil klasifikasi
prediction_count = {label: 0 for label in LABELS}

# ============================================================================
# PROSES PREDIKSI UNTUK SETIAP GAMBAR
# ============================================================================

print(f"\n🔍 Memindai folder '{INPUT_FOLDER}' untuk gambar...")

# Loop melalui semua file di folder input
for filename in os.listdir(INPUT_FOLDER):
    # Filter hanya file gambar dengan ekstensi yang didukung
    # .lower() untuk handle case sensitivity (PNG vs png)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Construct full path ke gambar
        img_path = os.path.join(INPUT_FOLDER, filename)
        
        # ====================================================================
        # PREPROCESS GAMBAR
        # ====================================================================
        
        # Buka gambar dengan PIL (Python Imaging Library)
        # .convert('L'): Convert ke grayscale (sesuai format Fashion MNIST)
        # .resize((28, 28)): Ubah ukuran ke 28x28 pixels (input model requirement)
        img = Image.open(img_path).convert('L').resize((28, 28))
        
        # ====================================================================
        # PREPARE INPUT UNTUK MODEL
        # ====================================================================
        
        # Konversi PIL Image ke numpy array
        # Nilai pixel: 0-255 (hitam-putih)
        img_array = np.array(img)
        
        # Normalisasi: ubah range [0, 255] → [0, 1]
        # Harus sama dengan normalisasi saat training!
        img_array = img_array / 255.0
        
        # Reshape untuk model: (28, 28) → (1, 28, 28)
        # Dimension: (batch_size, height, width)
        # batch_size=1 karena kita prediksi satu gambar per waktu
        img_array = img_array.reshape(1, 28, 28)
        
        # ====================================================================
        # PREDIKSI DENGAN MODEL
        # ====================================================================
        
        # Model.predict: melakukan forward propagation
        # Output: array 2D dengan shape (1, 10) - probabilitas 10 kelas
        # verbose=0: matikan progress bar (output lebih clean)
        prediction = model.predict(img_array, verbose=0)
        
        # Ambil index dengan probabilitas tertinggi
        # np.argmax: return index of maximum value
        pred_index = np.argmax(prediction)
        
        # Konversi index ke label text
        pred_label = LABELS[pred_index]
        
        # Hitung confidence score (probabilitas maksimal)
        # Convert ke percent dengan × 100
        confidence = float(np.max(prediction)) * 100
        
        # ====================================================================
        # OUTPUT DAN PENYIMPANAN HASIL
        # ====================================================================
        
        # Tampilkan hasil di terminal
        print(f"   📸 {filename}: {pred_label} ({confidence:.2f}%)")
        
        # Simpan gambar ke folder sesuai label prediksi
        output_path = os.path.join(OUTPUT_FOLDER, pred_label, filename)
        img.save(output_path)
        
        # Update statistik prediksi
        prediction_count[pred_label] += 1

# ============================================================================
# VISUALISASI DISTRIBUSI PREDIKSI
# ============================================================================

print(f"\n📊 Membuat visualisasi distribusi prediksi...")

# Extract data untuk plotting
labels = list(prediction_count.keys())
values = list(prediction_count.values())

# Buat figure dan axis untuk plotting
plt.figure(figsize=(10, 5))

# Buat bar chart
bars = plt.bar(labels, values, color='skyblue', edgecolor='navy', alpha=0.7)

# Konfigurasi chart
plt.title("Distribusi Prediksi Fashion MNIST", fontsize=14, fontweight='bold')
plt.xlabel("Kategori Fashion", fontsize=12)
plt.ylabel("Jumlah Gambar", fontsize=12)
plt.xticks(rotation=45, ha='right')  # Rotasi label x untuk readability
plt.grid(axis='y', linestyle='--', alpha=0.6)  # Grid horizontal

# Tambahkan nilai di atas setiap bar
for bar in bars:
    yval = bar.get_height()
    # Text position: center of bar, slightly above height
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, int(yval),
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Adjust layout untuk prevent label cutoff
plt.tight_layout()

# Tampilkan plot
plt.show()

# ============================================================================
# SUMMARY FINAL
# ============================================================================

total_images = sum(prediction_count.values())
print(f"\n🎉 PROSES SELESAI!")
print(f"📈 Total gambar diproses: {total_images}")
print(f"📁 Hasil disimpan di folder: '{OUTPUT_FOLDER}/'")
print(f"📊 Distribusi prediksi:")
for label, count in prediction_count.items():
    if count > 0:
        print(f"   • {label}: {count} gambar")