"""
Fashion MNIST Model Unit Testing
=================================
Script untuk melakukan unit testing terhadap model yang sudah dilatih.
Testing mencakup evaluasi akurasi, loss, dan validasi output prediksi.

Framework: unittest (bawaan Python)

Dibuat oleh: Fathih Apriandi
"""

import unittest
import tensorflow as tf

# ============================================================================
# KONFIGURASI PATH MODEL
# ============================================================================

# Path ke file model yang sudah dilatih
# File ini harus sudah ada (hasil dari train_fashion_mnist_model.py)
PATH_MODEL = 'fashion_mnist_model.keras'

# ============================================================================
# CLASS UNIT TEST
# ============================================================================

class TestFashionMNISTModel(unittest.TestCase):
    """
    Class untuk testing model Fashion MNIST.
    Mewarisi unittest.TestCase untuk menggunakan assertion methods.
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Method ini dijalankan SEKALI sebelum semua test method.
        Digunakan untuk setup yang memakan waktu (load model, load data).
        
        @classmethod decorator membuat method ini milik class, bukan instance.
        cls parameter mengacu ke class itu sendiri (seperti self untuk instance).
        """
        
        # Load dataset Fashion MNIST dari Keras
        mnist = tf.keras.datasets.fashion_mnist
        
        # Kita hanya butuh test data untuk validasi model
        # _ (underscore) adalah konvensi Python untuk variabel yang tidak digunakan
        # Di sini kita abaikan training data karena hanya butuh test data
        (_, _), (cls.x_test, cls.y_test) = mnist.load_data()
        
        # Normalisasi pixel values ke range [0, 1]
        # Harus sama dengan normalisasi saat training untuk hasil konsisten
        # Jika tidak dinormalisasi, model akan memberikan prediksi yang salah
        cls.x_test = cls.x_test / 255.0
        
        # Load model yang sudah dilatih dari file
        # Model ini berisi arsitektur dan weights hasil training
        cls.model = tf.keras.models.load_model(PATH_MODEL)
        
        print(f"\n✅ Setup selesai: Model dan test data berhasil dimuat")
    
    def test_model_evaluate(self):
        """
        Test 1: Evaluasi performa model pada test dataset.
        
        Validasi yang dilakukan:
        1. Accuracy berada dalam range [0, 1]
        2. Loss bernilai non-negatif (≥ 0)
        """
        
        # Evaluasi model menggunakan 10.000 gambar test
        # verbose=0 mematikan output progress bar
        # Return: (loss_value, accuracy_value)
        loss, akurasi = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        # Assertion 1: Akurasi harus >= 0.0
        # assertGreaterEqual berarti "pastikan nilai >= threshold"
        self.assertGreaterEqual(
            akurasi, 
            0.0,
            msg="Akurasi tidak boleh negatif"
        )
        
        # Assertion 2: Akurasi harus <= 1.0
        # assertLessEqual berarti "pastikan nilai <= threshold"
        # Akurasi maksimal adalah 100% atau 1.0
        self.assertLessEqual(
            akurasi, 
            1.0,
            msg="Akurasi tidak boleh lebih dari 1.0"
        )
        
        # Assertion 3: Loss harus >= 0.0
        # Loss function tidak boleh menghasilkan nilai negatif
        self.assertGreaterEqual(
            loss, 
            0.0,
            msg="Loss tidak boleh negatif"
        )
        
        print(f"   📊 Test Evaluate - Loss: {loss:.4f}, Accuracy: {akurasi:.4f}")
    
    def test_model_predict_shape(self):
        """
        Test 2: Validasi shape dan nilai output prediksi model.
        
        Validasi yang dilakukan:
        1. Output shape harus (10, 10) untuk 10 gambar input
        2. Setiap row (prediksi per gambar) harus sum = 1.0 (properti softmax)
        3. Semua probabilitas harus dalam range [0, 1]
        """
        
        # Prediksi 10 gambar pertama dari test dataset
        # Kita ambil 10 sample untuk testing (cukup dan cepat)
        # Output: array 2D dengan shape (10, 10)
        # Baris = gambar, Kolom = probabilitas untuk setiap kelas (0-9)
        prediksi = self.model.predict(self.x_test[:10])
        
        # Assertion 1: Shape output harus (10, 10)
        # 10 gambar input -> 10 baris output
        # 10 kelas Fashion MNIST -> 10 kolom output
        self.assertEqual(
            prediksi.shape, 
            (10, 10),
            msg=f"Shape prediksi salah. Expected: (10, 10), Got: {prediksi.shape}"
        )
        
        # Assertion 2: Setiap row harus sum = 1.0 (properti softmax)
        # Softmax memastikan output adalah distribusi probabilitas
        # Loop setiap baris (setiap prediksi gambar)
        for idx_baris, baris in enumerate(prediksi):
            # assertAlmostEqual untuk perbandingan float (toleransi pembulatan)
            # places=3 berarti toleransi sampai 3 digit desimal (0.001)
            self.assertAlmostEqual(
                sum(baris), 
                1.0, 
                places=3,
                msg=f"Sum probabilitas baris {idx_baris} bukan 1.0: {sum(baris)}"
            )
        
        # Assertion 3: Semua probabilitas harus dalam range [0, 1]
        # Probabilitas tidak boleh negatif atau lebih dari 1
        # .all() memastikan semua elemen array memenuhi kondisi
        kondisi_valid = ((prediksi >= 0.0) & (prediksi <= 1.0)).all()
        self.assertTrue(
            kondisi_valid,
            msg="Ada probabilitas yang di luar range [0, 1]"
        )
        
        print(f"   📊 Test Predict - Shape: {prediksi.shape}, All validations passed")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Block ini dijalankan hanya jika script dieksekusi langsung
# Tidak dijalankan jika script di-import sebagai module
if __name__ == "__main__":
    # Menjalankan semua test yang ada di class TestFashionMNISTModel
    # unittest.main() secara otomatis menemukan semua method yang dimulai dengan "test_"
    unittest.main()