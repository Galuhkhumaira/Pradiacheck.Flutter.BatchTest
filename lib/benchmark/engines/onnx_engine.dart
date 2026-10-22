// ===== Letakkan di: lib/benchmark/engines/onnx_engine.dart =====
// Paket: onnxruntime

import 'dart:io';

import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart' as ort;

import '../inference.dart';

class OnnxEngine implements InferenceEngine {
  @override
  String get engineName => 'ONNX Runtime';

  @override
  String get format => 'onnx';

  String _info = '';
  @override
  String get modelInfo => _info;

  ort.OrtSession? _session;
  ort.OrtSessionOptions? _options;
  ort.OrtRunOptions? _runOptions;
  String _inputName = '';

  @override
  Future<void> load(String modelPath) async {
    ort.OrtEnv.instance.init();
    // Opsi default (tanpa set jumlah thread), sama dengan versi Kotlin.
    final options = ort.OrtSessionOptions();
    _options = options;
    final bytes = await File(modelPath).readAsBytes();
    final session = ort.OrtSession.fromBuffer(bytes, options);
    _session = session;
    _inputName = session.inputNames.first;
    _runOptions = ort.OrtRunOptions();
    _info = 'Input ${session.inputNames} | Output ${session.outputNames}';
  }

  @override
  Future<Timed> predictTimed(img.Image image) async {
    final session = _session;
    if (session == null) throw StateError('Panggil load() dulu');
    final size = BenchConfig.inputSize;
    final sw = Stopwatch()..start();

    // 1) Preprocess (export ONNX Ultralytics memakai NCHW)
    var t = sw.elapsedMicroseconds;
    final data = preprocessToFloat(image, size, nchw: true);
    final input = ort.OrtValueTensor.createTensorWithDataList(data, [1, 3, size, size]);
    final preMs = msSince(sw, t);

    List<ort.OrtValue?>? outputs;
    try {
      // 2) Inference (+ ratakan output bersarang menjadi Float32List)
      t = sw.elapsedMicroseconds;
      outputs = session.run(_runOptions!, {_inputName: input});
      final flat = flattenNested(outputs[0]?.value);
      final infMs = msSince(sw, t);

      // 3) Postprocess (decode + NMS)
      t = sw.elapsedMicroseconds;
      final pred = decodeDetections(flat.data, shape: flat.shape);
      final postMs = msSince(sw, t);

      return Timed(pred, preMs, infMs, postMs);
    } finally {
      input.release();
      outputs?.forEach((e) => e?.release());
    }
  }

  @override
  Future<void> close() async {
    _runOptions?.release();
    _options?.release();
    _session?.release();
    _runOptions = null;
    _options = null;
    _session = null;
  }
}
