// ===== Letakkan di: lib/benchmark/inference.dart =====
//
// Bagian bersama untuk semua engine (TFLite, ONNX, ExecuTorch):
// konfigurasi, preprocessing, decode output YOLO + NMS.
// Logikanya sengaja dibuat sama dengan versi Kotlin supaya hasilnya sebanding.

import 'dart:typed_data';

import 'package:image/image.dart' as img;

class BenchConfig {
  static const int inputSize = 640;

  /// Urutan HARUS sama dengan names di data.yaml (id kelas di file label .txt).
  static const List<String> labels = ['Diabetes', 'Non-diabetes', 'Pradiabetes'];

  /// Sama dengan default Ultralytics (conf=0.25, iou=0.7), sama seperti versi Kotlin.
  static const double confThreshold = 0.25;
  static const double iouThreshold = 0.7;

  static const String framework = 'Flutter';

  /// Berapa kali seluruh dataset diulang (untuk stabilitas latency) dan jumlah warm-up.
  static const int repeats = 3;
  static const int warmup = 5;

  /// Nama yang dicetak di kolom "Model" pada CSV (sama seperti di sheet Test).
  static String displayName(String modelName) {
    switch (modelName) {
      case 'best8n':
        return 'YOLOv8n';
      case 'best11n':
        return 'YOLO11n';
      case 'best11s':
        return 'YOLO11s';
      default:
        return modelName;
    }
  }
}

class Prediction {
  final String label;
  final double confidence;
  final List<double> scores; // skor tertinggi tiap kelas
  final int detections; // jumlah kotak setelah NMS

  const Prediction(this.label, this.confidence, this.scores, this.detections);
}

/// Hasil satu inferensi beserta waktu tiap tahap (milidetik).
class Timed {
  final Prediction prediction;
  final double preMs; // resize + konversi ke tensor input
  final double infMs; // menjalankan model + menyalin output ke Float32List
  final double postMs; // decode + NMS + pemilihan kelas

  const Timed(this.prediction, this.preMs, this.infMs, this.postMs);
}

/// Kontrak yang sama untuk ketiga engine.
abstract class InferenceEngine {
  String get engineName;

  /// Ekstensi file model: 'tflite', 'onnx', atau 'pte'.
  String get format;

  /// Info diagnostik (shape/tipe tensor) untuk dicetak di laporan. Opsional.
  String get modelInfo => '';

  Future<void> load(String modelPath);
  Future<Timed> predictTimed(img.Image image);
  Future<void> close();
}

double msSince(Stopwatch sw, int fromMicros) =>
    (sw.elapsedMicroseconds - fromMicros) / 1000.0;

/// Preprocessing: resize ke size x size, RGB, nilai 0..1.
/// nchw = true  -> [1, 3, H, W]
/// nchw = false -> [1, H, W, 3]
Float32List preprocessToFloat(img.Image src, int size, {required bool nchw}) {
  final resized = (src.width == size && src.height == size)
      ? src
      : img.copyResize(src,
          width: size, height: size, interpolation: img.Interpolation.linear);
  // getBytes(order: rgb) menghasilkan tepat 3 byte per piksel (alpha dibuang kalau ada).
  final bytes = resized.getBytes(order: img.ChannelOrder.rgb);
  final area = size * size;
  final out = Float32List(3 * area);
  if (nchw) {
    for (var i = 0; i < area; i++) {
      final p = i * 3;
      out[i] = bytes[p] / 255.0;
      out[area + i] = bytes[p + 1] / 255.0;
      out[2 * area + i] = bytes[p + 2] / 255.0;
    }
  } else {
    for (var i = 0; i < out.length; i++) {
      out[i] = bytes[i] / 255.0;
    }
  }
  return out;
}

class _Det {
  final double cx, cy, w, h, conf;
  final int cls;
  const _Det(this.cx, this.cy, this.w, this.h, this.conf, this.cls);
}

double _iou(_Det a, _Det b) {
  final ax1 = a.cx - a.w / 2, ay1 = a.cy - a.h / 2;
  final ax2 = a.cx + a.w / 2, ay2 = a.cy + a.h / 2;
  final bx1 = b.cx - b.w / 2, by1 = b.cy - b.h / 2;
  final bx2 = b.cx + b.w / 2, by2 = b.cy + b.h / 2;
  final iw = (ax2 < bx2 ? ax2 : bx2) - (ax1 > bx1 ? ax1 : bx1);
  final ih = (ay2 < by2 ? ay2 : by2) - (ay1 > by1 ? ay1 : by1);
  if (iw <= 0 || ih <= 0) return 0;
  final inter = iw * ih;
  final union = a.w * a.h + b.w * b.h - inter;
  return union <= 0 ? 0 : inter / union;
}

/// Output model deteksi YOLO (Ultralytics). Normalnya channels-first [1, 4+nc, N]:
/// 4 baris pertama = kotak (cx, cy, w, h), baris berikutnya = skor tiap kelas.
/// Kalau [shape] menunjukkan channels-last [1, N, 4+nc], data ditranspose dulu.
/// Langkah: kelas terbaik tiap anchor -> saring conf -> NMS per kelas.
/// Kelas prediksi = kelas pada kotak dengan confidence tertinggi.
Prediction decodeDetections(Float32List rawIn, {List<int>? shape}) {
  final labels = BenchConfig.labels;
  final nc = labels.length;
  final channels = 4 + nc;
  if (rawIn.length % channels != 0) {
    throw Exception('Ukuran output ${rawIn.length} tidak cocok untuk $nc kelas '
        '(harus kelipatan $channels). Cek jumlah kelas model.');
  }
  final n = rawIn.length ~/ channels;

  var raw = rawIn;
  if (shape != null &&
      shape.length == 3 &&
      shape[1] != channels &&
      shape[2] == channels) {
    // channels-last [1, N, channels] -> channels-first
    final t = Float32List(rawIn.length);
    for (var i = 0; i < n; i++) {
      for (var c = 0; c < channels; c++) {
        t[c * n + i] = rawIn[i * channels + c];
      }
    }
    raw = t;
  }

  // Skor tertinggi per kelas di seluruh anchor (untuk laporan)
  final best = List<double>.filled(nc, 0.0);
  for (var c = 0; c < nc; c++) {
    final base = (4 + c) * n;
    var m = 0.0;
    for (var i = 0; i < n; i++) {
      final v = raw[base + i];
      if (v > m) m = v;
    }
    best[c] = m;
  }

  // Kandidat kotak
  final cands = <_Det>[];
  for (var i = 0; i < n; i++) {
    var bc = 0;
    var bs = raw[4 * n + i].toDouble();
    for (var c = 1; c < nc; c++) {
      final s = raw[(4 + c) * n + i].toDouble();
      if (s > bs) {
        bs = s;
        bc = c;
      }
    }
    if (bs >= BenchConfig.confThreshold) {
      cands.add(_Det(raw[i].toDouble(), raw[n + i].toDouble(),
          raw[2 * n + i].toDouble(), raw[3 * n + i].toDouble(), bs, bc));
    }
  }
  cands.sort((a, b) => b.conf.compareTo(a.conf));

  // NMS per kelas
  final kept = <_Det>[];
  for (final d in cands) {
    var suppressed = false;
    for (final k in kept) {
      if (k.cls == d.cls && _iou(k, d) > BenchConfig.iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.add(d);
  }

  if (kept.isEmpty) {
    final maxScore = best.reduce((a, b) => a > b ? a : b);
    return Prediction('tidak terdeteksi', maxScore, best, 0);
  }
  return Prediction(labels[kept.first.cls], kept.first.conf, best, kept.length);
}

/// Meratakan list bersarang (mis. [1][7][8400]) menjadi Float32List, sekaligus mengembalikan shape-nya.
({Float32List data, List<int> shape}) flattenNested(Object? value) {
  final shape = <int>[];
  Object? cur = value;
  while (cur is List) {
    shape.add(cur.length);
    if (cur.isEmpty) break;
    cur = cur.first;
  }
  var total = 1;
  for (final s in shape) {
    total *= s;
  }
  final out = Float32List(total);
  var pos = 0;
  void walk(Object? v) {
    if (v is List) {
      for (final e in v) {
        walk(e);
      }
    } else if (v is num) {
      out[pos++] = v.toDouble();
    }
  }

  walk(value);
  return (data: out, shape: shape);
}
