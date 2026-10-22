// ===== Letakkan di: lib/benchmark/batch_runner.dart =====
//
// Struktur folder di HP (milik aplikasi, tanpa permission), di dalam
// /sdcard/Android/data/<applicationId>/files/ :
//   models/best8n.tflite, best11n.onnx, best11s.pte, ...   <- semua model yang mau diuji
//   test/images/xxx.jpg                                    <- gambar uji
//   test/labels/xxx.txt                                    <- label YOLO: "class cx cy w h"
//   results/flutter_<model>_<format>/                      <- hasil, 1 folder per pengujian
//
// Pilih model dari dropdown di aplikasi: TIDAK perlu mengubah kode untuk ganti model/format.

import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';

import 'engines/executorch_engine.dart';
import 'engines/onnx_engine.dart';
import 'engines/tflite_engine.dart';
import 'inference.dart';

/// Satu file model di folder models/.
class ModelEntry {
  final String name; // mis. best8n
  final String format; // tflite | onnx | pte
  final String path;
  final int sizeBytes;

  const ModelEntry(this.name, this.format, this.path, this.sizeBytes);

  String get fileName => '$name.$format';
  String get display =>
      '${BenchConfig.displayName(name)} • $format  ($fileName, ${(sizeBytes / 1048576).toStringAsFixed(1)} MB)';
}

/// Pengukuran native (memori PSS & ukuran APK) lewat MethodChannel ke MainActivity.kt.
/// PSS sengaja dipakai (bukan RSS) supaya sebanding dengan versi Kotlin.
class NativeProbe {
  static const MethodChannel _ch = MethodChannel('bench/native');

  static Future<double> pssMb() async {
    try {
      return (await _ch.invokeMethod<double>('pssMb')) ?? -1.0;
    } catch (_) {
      return -1.0; // channel belum dipasang di MainActivity.kt
    }
  }

  static Future<double> apkSizeMb() async {
    try {
      return (await _ch.invokeMethod<double>('apkSizeMb')) ?? -1.0;
    } catch (_) {
      return -1.0;
    }
  }
}

class BatchSummary {
  final int total;
  final int correct;
  final String report;
  final String outDir;
  const BatchSummary(this.total, this.correct, this.report, this.outDir);
}

class BenchPaths {
  final Directory root;
  const BenchPaths(this.root);

  Directory get models => Directory('${root.path}/models');
  Directory get testImages => Directory('${root.path}/test/images');
  Directory get testLabels => Directory('${root.path}/test/labels');
  Directory get results => Directory('${root.path}/results');

  static Future<BenchPaths> resolve() async {
    final dir = await getExternalStorageDirectory();
    if (dir == null) {
      throw Exception('Penyimpanan eksternal aplikasi tidak tersedia di perangkat ini.');
    }
    return BenchPaths(dir);
  }
}

/// Daftar model yang ada di folder models/ (diurutkan berdasarkan nama).
Future<List<ModelEntry>> scanModels(BenchPaths paths) async {
  final dir = paths.models;
  if (!await dir.exists()) return [];
  final out = <ModelEntry>[];
  await for (final e in dir.list()) {
    if (e is! File) continue;
    final fileName = e.uri.pathSegments.last;
    final dot = fileName.lastIndexOf('.');
    if (dot <= 0) continue;
    final ext = fileName.substring(dot + 1).toLowerCase();
    if (ext != 'tflite' && ext != 'onnx' && ext != 'pte') continue;
    out.add(ModelEntry(fileName.substring(0, dot), ext, e.path, await e.length()));
  }
  out.sort((a, b) => a.fileName.compareTo(b.fileName));
  return out;
}

InferenceEngine createEngine(String format) {
  switch (format) {
    case 'tflite':
      return TfliteEngine();
    case 'onnx':
      return OnnxEngine();
    case 'pte':
      return ExecuTorchEngine();
    default:
      throw Exception('Format tidak dikenal: $format');
  }
}

// ---------- helper statistik & format ----------
String _f(double x, [int d = 2]) => x.toStringAsFixed(d);
double _mean(List<double> v) => v.isEmpty ? 0 : v.reduce((a, b) => a + b) / v.length;
double _median(List<double> v) {
  final s = [...v]..sort();
  return s[s.length ~/ 2];
}

double _std(List<double> v) {
  if (v.length < 2) return 0;
  final m = _mean(v);
  var s = 0.0;
  for (final x in v) {
    s += (x - m) * (x - m);
  }
  return math.sqrt(s / (v.length - 1));
}

String _two(int n) => n.toString().padLeft(2, '0');
String _stamp() {
  final d = DateTime.now();
  return '${d.year}-${_two(d.month)}-${_two(d.day)} ${_two(d.hour)}:${_two(d.minute)}:${_two(d.second)}';
}

/// Jalankan benchmark untuk SATU model. Hasil ditulis ke results/flutter_<model>_<format>/
/// (isi folder itu dihapus dulu, jadi hanya berisi hasil pengujian ini).
Future<BatchSummary> runBatch(
  BenchPaths paths,
  ModelEntry entry, {
  int repeats = BenchConfig.repeats,
  void Function(String status)? onProgress,
}) async {
  final labels = BenchConfig.labels;
  final cols = [...labels, 'tidak terdeteksi'];

  // ---- Dataset: pasangan gambar + label YOLO ----
  if (!await paths.testImages.exists() || !await paths.testLabels.exists()) {
    throw Exception('Folder test/images dan test/labels tidak ditemukan di ${paths.root.path}. '
        'Push dataset dulu pakai adb.');
  }
  final imageFiles = <File>[];
  await for (final e in paths.testImages.list()) {
    if (e is! File) continue;
    final n = e.path.toLowerCase();
    if (n.endsWith('.jpg') || n.endsWith('.jpeg') || n.endsWith('.png')) imageFiles.add(e);
  }
  imageFiles.sort((a, b) => a.path.compareTo(b.path));

  final names = <String>[]; // nama file gambar
  final paths_ = <String>[];
  final truths = <String>[];
  for (final f in imageFiles) {
    final fileName = f.uri.pathSegments.last;
    final dot = fileName.lastIndexOf('.');
    final stem = dot > 0 ? fileName.substring(0, dot) : fileName;
    final txt = File('${paths.testLabels.path}/$stem.txt');
    if (!await txt.exists()) continue;
    final lines = (await txt.readAsLines()).where((l) => l.trim().isNotEmpty).toList();
    if (lines.isEmpty) continue;
    final classId = int.tryParse(lines.first.trim().split(RegExp(r'\s+')).first);
    if (classId == null || classId < 0 || classId >= labels.length) continue;
    names.add(fileName);
    paths_.add(f.path);
    truths.add(labels[classId]);
  }
  final n = names.length;
  if (n == 0) {
    throw Exception('Tidak ada pasangan gambar+label yang valid. Cek nama file images/ dan labels/.');
  }
  final skipped = imageFiles.length - n;

  // ---- Memuat model + ukur RAM ----
  onProgress?.call('Memuat model ${entry.fileName}...');
  final ramBefore = await NativeProbe.pssMb();
  final loadSw = Stopwatch()..start();
  final engine = createEngine(entry.format);
  await engine.load(entry.path);
  final loadMs = loadSw.elapsedMicroseconds / 1000.0;
  final ramAfterLoad = await NativeProbe.pssMb();
  final apkSizeMb = await NativeProbe.apkSizeMb();
  final modelSizeMb = entry.sizeBytes / 1048576.0;

  try {
    Future<img.Image> loadImage(int i) async {
      final bytes = await File(paths_[i]).readAsBytes();
      final decoded = img.decodeImage(bytes);
      if (decoded == null) throw Exception('Gagal membaca gambar: ${names[i]}');
      return decoded;
    }

    // ---- Cold start + warm-up ----
    onProgress?.call('Warm-up...');
    final first = await loadImage(0);
    final cold = await engine.predictTimed(first);
    final coldMs = cold.preMs + cold.infMs + cold.postMs;
    for (var i = 0; i < BenchConfig.warmup; i++) {
      await engine.predictTimed(first);
    }

    // ---- Loop utama ----
    final preA = List.generate(repeats, (_) => List<double>.filled(n, 0));
    final infA = List.generate(repeats, (_) => List<double>.filled(n, 0));
    final postA = List.generate(repeats, (_) => List<double>.filled(n, 0));
    final e2eA = List.generate(repeats, (_) => List<double>.filled(n, 0));
    final run1 = List<Prediction?>.filled(n, null);
    var peak = ramAfterLoad;
    final clock = Stopwatch()..start();

    for (var r = 0; r < repeats; r++) {
      for (var i = 0; i < n; i++) {
        final e0 = clock.elapsedMicroseconds;
        final image = await loadImage(i);
        final t = await engine.predictTimed(image);
        final e1 = clock.elapsedMicroseconds;

        preA[r][i] = t.preMs;
        infA[r][i] = t.infMs;
        postA[r][i] = t.postMs;
        e2eA[r][i] = (e1 - e0) / 1000.0;
        if (r == 0) run1[i] = t.prediction;

        // Di luar timer: ukur RAM puncak dan beri kesempatan UI memperbarui progres.
        final ram = await NativeProbe.pssMb();
        if (ram > peak) peak = ram;
        if (i % 10 == 0) {
          onProgress?.call('Ulangan ${r + 1}/$repeats • gambar ${i + 1}/$n');
          await Future<void>.delayed(Duration.zero);
        }
      }
    }

    // ---- Akurasi + confusion matrix (dari run pertama; deterministik) ----
    final confusion = List.generate(labels.length, (_) => List<int>.filled(cols.length, 0));
    var correct = 0;
    for (var i = 0; i < n; i++) {
      final pred = run1[i]!;
      final ti = labels.indexOf(truths[i]);
      final pi = cols.indexOf(pred.label);
      if (ti >= 0 && pi >= 0) confusion[ti][pi]++;
      if (pred.label == truths[i]) correct++;
    }
    final precs = <double>[], recs = <double>[], f1s = <double>[];
    final perClass = StringBuffer();
    for (var c = 0; c < labels.length; c++) {
      final tp = confusion[c][c].toDouble();
      var colSum = 0;
      for (var r = 0; r < labels.length; r++) {
        colSum += confusion[r][c];
      }
      final rowSum = confusion[c].reduce((a, b) => a + b);
      final fp = colSum - tp;
      final fn = rowSum - tp;
      final p = tp + fp > 0 ? tp / (tp + fp) : 0.0;
      final rc = tp + fn > 0 ? tp / (tp + fn) : 0.0;
      final f1 = p + rc > 0 ? 2 * p * rc / (p + rc) : 0.0;
      precs.add(p);
      recs.add(rc);
      f1s.add(f1);
      perClass.writeln('${labels[c]}: P=${_f(p, 3)} R=${_f(rc, 3)} F1=${_f(f1, 3)}');
    }

    // ---- Statistik waktu ----
    final totals = List.generate(
        repeats, (r) => List.generate(n, (i) => preA[r][i] + infA[r][i] + postA[r][i]));
    final totalFlat = totals.expand((x) => x).toList();
    final runMeans = totals.map(_mean).toList();
    final preAvg = _mean(preA.expand((x) => x).toList());
    final infAvg = _mean(infA.expand((x) => x).toList());
    final postAvg = _mean(postA.expand((x) => x).toList());
    final e2eAvg = _mean(e2eA.expand((x) => x).toList());
    final totAvg = _mean(totalFlat);
    final totMedian = _median(totalFlat);
    final totStd = _std(totalFlat);
    final runStd = _std(runMeans);
    final totMin = totalFlat.reduce(math.min);
    final totMax = totalFlat.reduce(math.max);

    final stamp = _stamp();
    final modelDisplay = BenchConfig.displayName(entry.name);
    final engineTag = engine.engineName;

    // ---- CSV detail (kolom sama dengan sheet "Test") ----
    double avgAt(List<List<double>> a, int i) =>
        _mean(List.generate(repeats, (r) => a[r][i]));
    final detail = StringBuffer(
        'Model,Format,Gambar,Label Asli,Kelas Prediksi,Cocok,Jumlah Deteksi,Confidence,'
        'Waktu Preprocess (ms),Waktu Inference (ms),Waktu Postprocess (ms),'
        'Waktu Total (ms),Waktu End-to-End (ms),Waktu Tes,Framework,Engine\n');
    for (var i = 0; i < n; i++) {
      final p = run1[i]!;
      final pre = avgAt(preA, i), inf = avgAt(infA, i), post = avgAt(postA, i);
      detail.write('$modelDisplay,${entry.format},${names[i]},${truths[i]},${p.label},'
          '${p.label == truths[i] ? 'Ya' : 'Tidak'},${p.detections},${_f(p.confidence, 4)},'
          '${_f(pre)},${_f(inf)},${_f(post)},${_f(pre + inf + post)},${_f(avgAt(e2eA, i))},'
          '$stamp,${BenchConfig.framework},$engineTag\n');
    }

    // ---- CSV ringkasan ----
    const sumHeader = 'Framework,Engine,Model,Format,Jumlah Gambar,Ulangan,Accuracy (%),'
        'Precision Macro,Recall Macro,F1 Macro,'
        'Preprocess Avg (ms),Inference Avg (ms),Postprocess Avg (ms),Total Avg (ms),Total Median (ms),'
        'Total Std (ms),Std Antar Ulangan (ms),Total Min (ms),Total Max (ms),End-to-End Avg (ms),'
        'Cold Start (ms),Model Load (ms),Ukuran Model (MB),Ukuran APK (MB),'
        'RAM Sebelum Load (MB),RAM Setelah Load (MB),RAM Puncak (MB),Tambahan RAM Puncak (MB),Waktu Tes\n';
    final sumRow = [
      BenchConfig.framework, engineTag, modelDisplay, entry.format,
      '$n', '$repeats', _f(100.0 * correct / n),
      _f(_mean(precs), 4), _f(_mean(recs), 4), _f(_mean(f1s), 4),
      _f(preAvg), _f(infAvg), _f(postAvg), _f(totAvg), _f(totMedian),
      _f(totStd), _f(runStd), _f(totMin), _f(totMax), _f(e2eAvg),
      _f(coldMs), _f(loadMs), _f(modelSizeMb), _f(apkSizeMb),
      _f(ramBefore), _f(ramAfterLoad), _f(peak), _f(peak - ramBefore), stamp,
    ].join(',') + '\n';

    // ---- Laporan teks ----
    final sb = StringBuffer();
    sb.writeln('Model: ${entry.name} ($modelDisplay) | Format: ${entry.format} | Engine: $engineTag | Framework: ${BenchConfig.framework}');
    if (engine.modelInfo.isNotEmpty) sb.writeln('Tensor: ${engine.modelInfo}');
    sb.writeln('Gambar: $n | Ulangan: $repeats | Pengukuran: ${totalFlat.length}');
    if (skipped > 0) sb.writeln('PERINGATAN: $skipped gambar dilewati (label .txt tidak ada/kosong)');
    sb.writeln();
    sb.writeln('Accuracy: $correct/$n = ${_f(100.0 * correct / n)}%');
    sb.writeln('Precision/Recall/F1 (macro): ${_f(_mean(precs), 3)} / ${_f(_mean(recs), 3)} / ${_f(_mean(f1s), 3)}');
    sb.writeln();
    sb.writeln('Waktu rata-rata (ms): pre=${_f(preAvg)} inf=${_f(infAvg)} post=${_f(postAvg)} total=${_f(totAvg)}');
    sb.writeln('Total: median=${_f(totMedian)} std=${_f(totStd)} min=${_f(totMin)} max=${_f(totMax)}');
    sb.writeln('Rata-rata tiap ulangan (ms): ${runMeans.map(_f).join(' / ')}');
    sb.writeln('Std antar ulangan: ${_f(runStd)} | End-to-end avg: ${_f(e2eAvg)}');
    sb.writeln('Cold start: ${_f(coldMs)} ms | Model load: ${_f(loadMs)} ms');
    sb.writeln();
    sb.writeln('Ukuran model: ${_f(modelSizeMb)} MB | Ukuran APK: ${apkSizeMb < 0 ? 'N/A' : _f(apkSizeMb)} MB');
    if (ramBefore < 0) {
      sb.writeln('RAM: tidak tersedia (MethodChannel belum dipasang di MainActivity.kt)');
    } else {
      sb.writeln('RAM (PSS): sebelum load=${_f(ramBefore)} setelah load=${_f(ramAfterLoad)} puncak=${_f(peak)} MB');
    }
    sb.writeln();
    sb.writeln('Confusion matrix (baris=truth, kolom=pred: ${cols.join(', ')}):');
    for (var i = 0; i < labels.length; i++) {
      sb.writeln('${labels[i]}: ${confusion[i].join(', ')}');
    }
    sb.writeln();
    sb.writeln('Per kelas:');
    sb.write(perClass.toString());

    // ---- Tulis file: 1 folder per pengujian, dikosongkan dulu ----
    final base = 'flutter_${entry.name}_${entry.format}';
    final outDir = Directory('${paths.results.path}/$base');
    if (await outDir.exists()) await outDir.delete(recursive: true);
    await outDir.create(recursive: true);
    await File('${outDir.path}/${base}_detail.csv').writeAsString(detail.toString());
    await File('${outDir.path}/${base}_ringkasan.csv').writeAsString(sumHeader + sumRow);
    await File('${outDir.path}/$base.txt').writeAsString(sb.toString());

    return BatchSummary(n, correct, sb.toString(), outDir.path);
  } finally {
    await engine.close();
  }
}
