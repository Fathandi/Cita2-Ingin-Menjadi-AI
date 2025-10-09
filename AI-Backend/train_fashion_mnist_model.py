"""
Fashion MNIST Model Training Script
====================================
Script untuk melatih model neural network pada dataset Fashion MNIST.
Model akan dikomputasi selama 10 epochs dan disimpan dalam format .keras

Arsitektur Model:
- Input: Flatten layer (28x28 pixels → 784 neurons)
- Hidden: Dense layer 512 neurons dengan aktivasi ReLU
- Output: Dense layer 10 neurons dengan aktivasi Softmax

Optimizer: Adam
Loss Function: Sparse Categorical Crossentropy
Metrics: Accuracy

Dibuat oleh: Fathih Apriandi
"""

import tensorflow as tf

# ============================================================================
# LOAD DAN PREPROCESS DATASET FASHION MNIST
# ============================================================================

# Load dataset Fashion MNIST dari TensorFlow Keras
# Dataset berisi 70.000 gambar fashion item dalam 10 kategori
# - Training set: 60.000 gambar (x_train, y_train)
# - Test set: 10.000 gambar (x_test, y_test)
# Pixel values: 0-255 (grayscale), Label: 0-9
mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisasi pixel values dari range [0, 255] ke [0, 1]
# Tujuan: 
# 1. Mempercepat konvergensi selama training
# 2. Mencegah gradient explosion
# 3. Membuat model lebih stabil secara numerik
x_train, x_test = x_train / 255.0, x_test / 255.0

# ============================================================================
# ARSITEKTUR MODEL NEURAL NETWORK
# ============================================================================

# Membangun model Sequential (feed-forward neural network)
# Sequential model: layer disusun secara berurutan dari input ke output
model = tf.keras.models.Sequential([
    # Layer 1: Flatten
    # Mengubah input 2D (28x28) menjadi 1D vector (784 elements)
    # Required karena dense layer menerima input 1D
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    
    # Layer 2: Dense (Hidden Layer)
    # 512 neurons dengan aktivasi ReLU (Rectified Linear Unit)
    # ReLU: f(x) = max(0, x) - menghilangkan nilai negatif
    # 512 neurons memberikan kapasitas learning yang cukup tanpa overfitting berlebihan
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    
    # Layer 3: Dense (Output Layer)
    # 10 neurons (sesuai jumlah kelas Fashion MNIST)
    # Aktivasi Softmax: mengubah output menjadi probabilitas (sum = 1.0)
    # Setiap neuron mewakili probabilitas satu kelas fashion item
    tf.keras.layers.Dense(10, activation='softmax')
])

# ============================================================================
# KOMPILASI MODEL
# ============================================================================

# Kompilasi model dengan konfigurasi training
model.compile(
    # Optimizer: Adam (Adaptive Moment Estimation)
    # Kombinasi RMSProp + Momentum, cocok untuk berbagai masalah
    # Learning rate adaptive, tidak perlu manual tuning
    optimizer=tf.optimizers.Adam(),
    
    # Loss Function: Sparse Categorical Crossentropy
    # Cocok untuk multi-class classification dengan integer labels
    # Tidak perlu one-hot encoding untuk target labels
    loss='sparse_categorical_crossentropy',
    
    # Metrics: Accuracy
    # Mengukur persentase prediksi yang benar
    # Monitor utama selama training process
    metrics=['accuracy']
)

# ============================================================================
# TRAINING PROCESS
# ============================================================================

# Melatih model dengan data training
# Parameter:
# - x_train: Gambar training (60.000 samples)
# - y_train: Label training (0-9)
# - epochs: 10 (seluruh dataset diproses 10 kali)
# 
# Proses training:
# 1. Forward propagation: input → output
# 2. Calculate loss: bandingkan prediksi vs actual
# 3. Backward propagation: hitung gradients
# 4. Update weights: adjust berdasarkan gradients
print("🚀 Memulai training model...")
history = model.fit(x_train, y_train, epochs=10)

# ============================================================================
# SIMPAN MODEL
# ============================================================================

# Menyimpan model yang sudah dilatih ke file .keras
# Format .keras: format modern TensorFlow, menyimpan:
# - Arsitektur model
# - Weight values
# - Optimizer state
# - Loss dan metrics
model.save('fashion_mnist_model.keras')

# ============================================================================
# EVALUASI FINAL
# ============================================================================

# Evaluasi model dengan test dataset
# Memberikan indikasi performa pada data yang belum pernah dilihat
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\n✅ Training selesai!")
print(f"📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"📊 Test Loss: {test_loss:.4f}")
print(f"💾 Model disimpan sebagai: fashion_mnist_model.keras")