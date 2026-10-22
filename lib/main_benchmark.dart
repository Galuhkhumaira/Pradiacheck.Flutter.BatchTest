// ===== Letakkan di: lib/main_benchmark.dart =====
// Entry point TERPISAH untuk benchmark, supaya aplikasi aslimu (lib/main.dart) tidak tersentuh.
// Jalankan dengan:  flutter run --release -t lib/main_benchmark.dart

import 'package:flutter/material.dart';

import 'benchmark/benchmark_screen.dart';

void main() {
  runApp(const MaterialApp(
    debugShowCheckedModeBanner: false,
    home: BenchmarkScreen(),
  ));
}
