import tensorflow as tf
import numpy as np
from PIL import Image
import os
import shutil
import matplotlib.pyplot as plt

# Label resmi Fashion MNIST
LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load model
model = tf.keras.models.load_model('fashion_mnist_model.keras')

# Folder input & output
INPUT_FOLDER = 'test-image'
OUTPUT_FOLDER = 'result'

# Bersihkan folder result kalau sudah ada
if os.path.exists(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER)

# Buat subfolder per label
for label in LABELS:
    os.makedirs(os.path.join(OUTPUT_FOLDER, label), exist_ok=True)

# Statistik prediksi
prediction_count = {label: 0 for label in LABELS}

# Loop semua gambar
for filename in os.listdir(INPUT_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(INPUT_FOLDER, filename)
        img = Image.open(img_path).convert('L').resize((28, 28))

        # Normalisasi & reshape ke input model
        img_array = np.array(img) / 255.0
        img_array = img_array.reshape(1, 28, 28)

        # Prediksi
        prediction = model.predict(img_array, verbose=0)
        pred_index = np.argmax(prediction)
        pred_label = LABELS[pred_index]
        confidence = float(np.max(prediction)) * 100

        # Tampilkan hasil di terminal
        print(f"{filename}: {pred_label} ({confidence:.2f}%)")

        # Simpan ke folder label hasil prediksi
        output_path = os.path.join(OUTPUT_FOLDER, pred_label, filename)
        img.save(output_path)

        # Tambahkan ke statistik
        prediction_count[pred_label] += 1

# Visualisasi hasil prediksi
labels = list(prediction_count.keys())
values = list(prediction_count.values())

plt.figure(figsize=(10, 5))
bars = plt.bar(labels, values)
plt.title("Distribusi Prediksi Fashion MNIST")
plt.xlabel("Label Prediksi")
plt.ylabel("Jumlah Gambar")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Tambah label angka di atas bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, int(yval),
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

print("\n✅ Semua gambar sudah diklasifikasikan ke folder 'result/'")
