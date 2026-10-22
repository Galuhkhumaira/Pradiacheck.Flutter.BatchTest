// ===== Letakkan di: lib/benchmark/engines/tflite_engine.dart =====
// Paket: tflite_flutter

import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import '../inference.dart';

class TfliteEngine implements InferenceEngine {
  @override
  String get engineName => 'TensorFlow Lite';

  @override
  String get format => 'tflite';

  String _info = '';
  @override
  String get modelInfo => _info;

  tfl.Interpreter? _interp;
  bool _nchw = true;
  List<int> _outShape = const [];

  @override
  Future<void> load(String modelPath) async {
    // 4 thread, sama dengan versi Kotlin.
    final options = tfl.InterpreterOptions()..threads = 4;
    final interp = tfl.Interpreter.fromFile(File(modelPath), options: options);
    _interp = interp;

    final inT = interp.getInputTensor(0);
    final outT = interp.getOutputTensor(0);
    final inShape = inT.shape;
    // [1, 3, H, W] -> NCHW, [1, H, W, 3] -> NHWC. Dideteksi dari shape, bukan diasumsikan.
    _nchw = inShape.length == 4 && inShape[1] == 3 && inShape[3] != 3;
    _outShape = outT.shape;
    _info = 'Input $inShape ${inT.type} | Output $_outShape ${outT.type} | Threads 4';
  }

  @override
  Future<Timed> predictTimed(img.Image image) async {
    final interp = _interp;
    if (interp == null) throw StateError('Panggil load() dulu');
    final sw = Stopwatch()..start();

    // 1) Preprocess
    var t = sw.elapsedMicroseconds;
    final data = preprocessToFloat(image, BenchConfig.inputSize, nchw: _nchw);
    final inputBytes = data.buffer.asUint8List();
    final preMs = msSince(sw, t);

    // 2) Inference (+ salin output ke Float32List)
    t = sw.elapsedMicroseconds;
    interp.runInference([inputBytes]);
    final view = interp.getOutputTensor(0).data; // view ke memori native
    final copy = Uint8List.fromList(view);
    final scores = copy.buffer.asFloat32List();
    final infMs = msSince(sw, t);

    // 3) Postprocess (decode + NMS)
    t = sw.elapsedMicroseconds;
    final pred = decodeDetections(scores, shape: _outShape);
    final postMs = msSince(sw, t);

    return Timed(pred, preMs, infMs, postMs);
  }

  @override
  Future<void> close() async {
    _interp?.close();
    _interp = null;
  }
}
