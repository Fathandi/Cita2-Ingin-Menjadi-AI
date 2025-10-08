# AI-Backend: Fashion MNIST Classification

## Deskripsi
Project ini adalah backend sederhana untuk klasifikasi gambar Fashion MNIST menggunakan TensorFlow.  
Terdapat dua opsi pengetesan model: otomatis dengan dataset resmi dan manual dengan gambar custom.

---

## Tech Stack

- **Python 3.12.3**
- **TensorFlow**: Framework deep learning untuk training dan inferensi model.
- **NumPy**: Manipulasi array dan data numerik.
- **Pillow (PIL)**: Proses dan konversi gambar.
- **Matplotlib**: Visualisasi hasil prediksi.
- **unittest**: Framework unit testing Python.
- **shutil & os**: Manajemen file dan folder.

---

## Workflow & Alur Program

1. **Ambil Gambar Sample**
   - Jalankan `download_image.py` untuk mengambil 100 gambar acak dari dataset Fashion MNIST dan menyimpannya ke folder `test-image/`.

2. **Training Model**
   - Jalankan `train_fashion_mnist_model.py` untuk melatih model neural network pada dataset Fashion MNIST.
   - Model hasil training disimpan sebagai `fashion_mnist_model.keras`.

3. **Pengetesan Model**
   - **Opsi 1: Otomatis (unit test)**
     - Jalankan `test_model.py` untuk menguji model menggunakan data test resmi Fashion MNIST.
     - Script ini akan mengecek akurasi dan output prediksi model secara otomatis.
   - **Opsi 2: Manual (custom image)**
     - Simpan gambar custom (grayscale, 28x28 piksel) ke folder `test-image/`.
     - Jalankan `predict_custom_image.py` untuk memprediksi gambar-gambar tersebut.
     - Hasil prediksi akan dikelompokkan ke subfolder di `result/` sesuai label prediksi, dan distribusi prediksi divisualisasikan dengan grafik.

---

## Cara Instalasi

1. **Clone repository**
   ```bash
   git clone https://github.com/Fathandi/Cita2-Ingin-Menjadi-AI.git
   cd AI-Backend
   ```

2. **Buat virtual environment (opsional)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install library yang dibutuhkan**
   ```bash
   pip install tensorflow pillow matplotlib numpy
   ```

---

## Cara Penggunaan

### 1. Download Gambar Sample
```bash
python download_image.py
```
Gambar akan tersimpan di folder `test-image/`.

### 2. Training Model
```bash
python train_fashion_mnist_model.py
```
Model hasil training akan tersimpan sebagai `fashion_mnist_model.keras`.

### 3. Pengetesan Model

#### Opsi 1: Otomatis (unit test)
```bash
python -m unittest test_model.py
```
atau dengan coverage:
```bash
coverage run test_model.py
```

#### Opsi 2: Manual (custom image)
- Simpan gambar custom (format PNG/JPG, ukuran 28x28 piksel, grayscale) ke folder `test-image/`.
- Jalankan:
  ```bash
  python predict_custom_image.py
  ```
- Hasil prediksi akan dikelompokkan ke subfolder di `result/` dan distribusi label divisualisasikan.

---

## Catatan
- Model hanya mengenali 10 kelas Fashion MNIST: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
- Untuk hasil optimal, pastikan gambar custom mirip dengan data Fashion MNIST (grayscale, 28x28 piksel).
- Untuk GPU support, pastikan CUDA sudah terinstall (opsional).

---

## Lisensi
Presented by Fathih Apriandi
