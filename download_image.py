import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random

# Buat folder tujuan
output_folder = "test-image"
os.makedirs(output_folder, exist_ok=True)

# Load dataset resmi dari Keras
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Gabung semua data
x_all = np.concatenate((x_train, x_test))

# Ambil 100 gambar acak dari keseluruhan dataset
chosen_indices = random.sample(range(len(x_all)), 100)

for i, idx in enumerate(chosen_indices):
    img_array = x_all[idx]
    img = Image.fromarray(img_array)
    img = img.convert("L")  # grayscale
    
    filename = f"sample_{i+1}.png"
    img.save(os.path.join(output_folder, filename))

print(f"Selesai! 100 gambar acak disimpan di folder '{output_folder}'")
