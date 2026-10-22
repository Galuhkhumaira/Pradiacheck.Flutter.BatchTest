// ===== Letakkan di: lib/benchmark/engines/executorch_engine.dart =====
// Paket: executorch_flutter (butuh Flutter 3.38+)

import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart' as et;
import 'package:image/image.dart' as img;

import '../inference.dart';

class ExecuTorchEngine implements InferenceEngine {
  @override
  String get engineName => 'ExecuTorch (PyTorch)';

  @override
  String get format => 'pte';

  String _info = '';
  @override
  String get modelInfo => _info;

  et.ExecuTorchModel? _model;

  @override
  Future<void> load(String modelPath) async {
    _model = await et.ExecuTorchModel.load(modelPath);
    _info = 'Backend default paket (XNNPACK di Android)';
  }

  @override
  Future<Timed> predictTimed(img.Image image) async {
    final model = _model;
    if (model == null) throw StateError('Panggil load() dulu');
    final size = BenchConfig.inputSize;
    final sw = Stopwatch()..start();

    // 1) Preprocess (NCHW: [1, 3, H, W])
    var t = sw.elapsedMicroseconds;
    final data = preprocessToFloat(image, size, nchw: true);
    final tensor = et.TensorData(
      shape: [1, 3, size, size],
      dataType: et.TensorType.float32,
      data: data.buffer.asUint8List(),
    );
    final preMs = msSince(sw, t);

    // 2) Inference (+ salin output ke Float32List)
    t = sw.elapsedMicroseconds;
    final outputs = await model.forward([tensor]);
    final out = outputs.first;
    final copy = Uint8List.fromList(out.data);
    final scores = copy.buffer.asFloat32List();
    final infMs = msSince(sw, t);

    // 3) Postprocess (decode + NMS)
    t = sw.elapsedMicroseconds;
    // Di executorch_flutter 0.8.0 shape bertipe List<int?>, jadi dibersihkan dulu.
    final outShape = out.shape.whereType<int>().toList();
    final pred = decodeDetections(scores, shape: outShape);
    final postMs = msSince(sw, t);

    if (_info.startsWith('Backend')) {
      _info = 'Input [1, 3, $size, $size] FLOAT32 | Output $outShape ${out.dataType}';
    }
    return Timed(pred, preMs, infMs, postMs);
  }

  @override
  Future<void> close() async {
    await _model?.dispose();
    _model = null;
  }
}