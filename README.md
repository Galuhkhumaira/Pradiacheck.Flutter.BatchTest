# Pradiacheck.Flutter.BatchTest

Aplikasi Flutter untuk **batch testing** deteksi dini diabetes dari citra lidah menggunakan model YOLO yang berjalan **offline di Android**. Repo ini adalah bagian dari Tugas Akhir (S1 Teknik Biomedis, Universitas Telkom) dan menjadi salah satu dari tiga aplikasi pembanding (Flutter, Kotlin, React Native).

Tujuan: membandingkan **3 model** x **3 konfigurasi engine** langsung di HP, tanpa mengubah kode di antara kombinasi, lalu memilih satu kombinasi terbaik untuk aplikasi final.

## Kombinasi yang diuji

| Model | ONNX Runtime (`.onnx`) | ExecuTorch (`.pte`) | TFLite (`.tflite`) |
|-------|:----------------------:|:-------------------:|:------------------:|
| YOLOv8n (`best8n`)   | ✓ | ✓ | ✓ |
| YOLO11n (`best11n`)  | ✓ | ✓ | ✓ |
| YOLO11s (`best11s`)  | ✓ | ✓ | ✓ |

Total 9 kombinasi per aplikasi.

## Kelas

Urutan kelas mengikuti `model.names`:

| Indeks | Kelas |
|:------:|-------|
| 0 | Diabetes |
| 1 | Nondiabetes |
| 2 | Pradiabetes |

## Dataset uji

94 gambar (format YOLO, dikonversi ke folder per kelas):

- Pradiabetes: 35
- Diabetes: 31
- Nondiabetes: 28

Dataset **tidak** disertakan di APK dan tidak ada di repo ini. Dataset dikirim ke HP lewat `adb push`.

## Alur pengujian

1. Pilih model dan engine dari layar benchmark di aplikasi (tanpa edit kode).
2. Jalankan batch test pada seluruh gambar di folder dataset di HP.
3. Protokol: 94 gambar x 3 kali pengulangan, sama dengan aplikasi Kotlin dan React Native.
4. Hasil ditulis ke file CSV, lalu diambil ke laptop dengan `adb pull`.

Waktu dicatat per tahap: preprocess, inferensi, postprocess, dan end-to-end (ms).

### Perintah adb

```bash
# kirim dataset ke HP
adb push <folder-dataset-lokal> <folder-tujuan-di-HP>

# ambil hasil ke laptop
adb pull <folder-hasil-di-HP> <folder-tujuan-lokal>
```

Sesuaikan path dengan yang dipakai aplikasi.

## Menjalankan

Prasyarat: Flutter SDK, Android SDK, dan HP Android dengan USB debugging aktif.

```bash
flutter pub get
flutter run                 # debug
flutter run --release       # release
```

## Struktur singkat

- `lib/benchmark/engines/` : implementasi engine (`onnx_engine.dart`, `tflite_engine.dart`, `executorch_engine.dart`)
- `lib/benchmark/` : layar benchmark dan pengatur batch test (`benchmark_screen.dart`, `batch_runner.dart`)
- `assets/` : file model

## Dependensi utama

`onnxruntime`, `tflite_flutter`, `executorch_flutter`, `image_picker`.

## Perangkat uji

Samsung Galaxy A56 (Android).

## Repo terkait

- Pradiacheck.RN.BatchTest
- Pradiacheck-Kotlin

## Catatan

Benchmark ini hanya untuk memilih kombinasi terbaik. Aplikasi final akan memakai satu model dengan satu format saja.
