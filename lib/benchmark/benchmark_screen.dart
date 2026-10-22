// ===== Letakkan di: lib/benchmark/benchmark_screen.dart =====

import 'dart:io';

import 'package:flutter/material.dart';

import 'batch_runner.dart';
import 'inference.dart';

class BenchmarkScreen extends StatefulWidget {
  const BenchmarkScreen({super.key});

  @override
  State<BenchmarkScreen> createState() => _BenchmarkScreenState();
}

class _BenchmarkScreenState extends State<BenchmarkScreen> {
  BenchPaths? _paths;
  List<ModelEntry> _models = [];
  ModelEntry? _selected;
  bool _running = false;
  String _status = '';
  String _report = '';
  String? _error;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    try {
      final paths = await BenchPaths.resolve();
      final models = await scanModels(paths);
      if (!mounted) return;
      setState(() {
        _paths = paths;
        _models = models;
        // Pertahankan pilihan kalau modelnya masih ada
        _selected = models.where((m) => m.path == _selected?.path).firstOrNull ??
            (models.isNotEmpty ? models.first : null);
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = '$e');
    }
  }

  Future<void> _run() async {
    final paths = _paths;
    final entry = _selected;
    if (paths == null || entry == null) return;
    setState(() {
      _running = true;
      _report = '';
      _error = null;
      _status = 'Mulai...';
    });
    try {
      final r = await runBatch(
        paths,
        entry,
        onProgress: (s) {
          if (mounted) setState(() => _status = s);
        },
      );
      if (!mounted) return;
      setState(() {
        _report = '${r.report}\nHasil disimpan di:\n${r.outDir}';
        _status = 'Selesai';
      });
    } catch (e, st) {
      if (!mounted) return;
      setState(() {
        _error = '$e\n$st';
        _status = 'Gagal';
      });
    } finally {
      if (mounted) setState(() => _running = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final paths = _paths;
    return Scaffold(
      appBar: AppBar(title: const Text('Benchmark (Flutter)')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: [
            Text('Model: dipilih dari folder models/ (tidak perlu ubah kode)',
                style: Theme.of(context).textTheme.bodySmall),
            const SizedBox(height: 4),
            SelectableText('Folder data:\n${paths?.root.path ?? '...'}',
                style: const TextStyle(fontSize: 12)),
            const SizedBox(height: 12),
            if (_models.isEmpty)
              Text(
                'Belum ada model di ${paths?.models.path ?? 'models/'}.\n'
                'Push model lewat adb (.tflite / .onnx / .pte), lalu tekan Muat ulang.',
                style: const TextStyle(color: Colors.red),
              )
            else
              DropdownButtonFormField<ModelEntry>(
                isExpanded: true,
                value: _selected,
                decoration: const InputDecoration(
                  labelText: 'Pilih model + format',
                  border: OutlineInputBorder(),
                ),
                items: [
                  for (final m in _models)
                    DropdownMenuItem(value: m, child: Text(m.display, overflow: TextOverflow.ellipsis)),
                ],
                onChanged: _running ? null : (m) => setState(() => _selected = m),
              ),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: FilledButton(
                    onPressed: (_running || _selected == null) ? null : _run,
                    child: Text(_running ? 'Berjalan...' : 'Run Benchmark'),
                  ),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: _running ? null : _refresh,
                  child: const Text('Muat ulang'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            OutlinedButton(
              // Menutup proses sepenuhnya supaya RAM dan runtime benar-benar bersih
              // sebelum pengujian berikutnya (buka aplikasi lagi setelah ini).
              onPressed: _running ? null : () => exit(0),
              child: const Text('Keluar (reset RAM sebelum uji berikutnya)'),
            ),
            const SizedBox(height: 12),
            if (_status.isNotEmpty) Text(_status),
            if (_error != null)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: SelectableText('ERROR: $_error', style: const TextStyle(color: Colors.red)),
              ),
            if (_report.isNotEmpty)
              Padding(
                padding: const EdgeInsets.only(top: 8),
                child: SelectableText(_report, style: const TextStyle(fontFamily: 'monospace', fontSize: 12)),
              ),
            const SizedBox(height: 16),
            Text(
              'Label: ${BenchConfig.labels.join(', ')} | Ulangan: ${BenchConfig.repeats}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}
