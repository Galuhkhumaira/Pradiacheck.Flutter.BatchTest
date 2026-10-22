package com.example.pradiacheck_ai

import android.os.Debug
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import java.io.File

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "bench/native")
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    // Memori proses (PSS) dalam MB, sama dengan yang diukur di versi Kotlin
                    "pssMb" -> {
                        val info = Debug.MemoryInfo()
                        Debug.getMemoryInfo(info)
                        result.success(info.totalPss / 1024.0)
                    }
                    // Ukuran APK terpasang dalam MB
                    "apkSizeMb" -> {
                        val size = File(applicationInfo.sourceDir).length() / 1048576.0
                        result.success(size)
                    }
                    else -> result.notImplemented()
                }
            }
    }
}