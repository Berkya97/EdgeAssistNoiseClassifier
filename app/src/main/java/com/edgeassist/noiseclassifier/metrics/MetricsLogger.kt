package com.edgeassist.noiseclassifier.metrics

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Edge vs Cloud karşılaştırması için tahmin başına metrik kaydedicisi.
 *
 * Singleton — uygulama açılışında [init] ile başlatılır, kapanışta [close] çağrılır.
 * Her [log] çağrısı CSV dosyasına bir satır ekler. Thread-safe (synchronized).
 *
 * Çıktı yolu:
 *   <externalFilesDir>/metrics/inference_log_<yyyyMMdd_HHmmss>.csv
 */
object MetricsLogger {

    private const val TAG = "NoiseClassifier"
    private const val FLUSH_EVERY_N = 5

    data class MetricRecord(
        val timestamp: Long,
        val mode: String,
        val adaptiveDecision: String?,
        val totalLatencyMs: Long,
        val melExtractionMs: Long,
        val inferenceMs: Long,
        val networkMs: Long,
        val rttMs: Long?,
        val bytesSent: Long,
        val bytesReceived: Long,
        val batteryPercent: Int,
        val networkType: String,
        val prediction: String,
        val rawPrediction: String,
        val confidence: Float,
        val entropy: Float?,
        val isSilence: Boolean,
        val isUnknown: Boolean
    )

    private val lock = Any()
    private var writer: BufferedWriter? = null
    private var currentFile: File? = null
    private var pendingSinceFlush = 0

    private val CSV_HEADER = listOf(
        "timestamp",
        "mode",
        "adaptive_decision",
        "total_latency_ms",
        "mel_extraction_ms",
        "inference_ms",
        "network_ms",
        "rtt_ms",
        "bytes_sent",
        "bytes_received",
        "battery_percent",
        "network_type",
        "prediction",
        "raw_prediction",
        "confidence",
        "entropy",
        "is_silence",
        "is_unknown"
    )

    /** Uygulama açılışında çağır. Yeni timestamp'li CSV dosyası oluşturur. */
    fun init(context: Context) {
        synchronized(lock) {
            try {
                close()
                val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
                val metricsDir = File(baseDir, "metrics").apply { mkdirs() }
                val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
                val file = File(metricsDir, "inference_log_$ts.csv")
                val w = BufferedWriter(FileWriter(file, false))
                w.write(CSV_HEADER.joinToString(","))
                w.newLine()
                w.flush()
                writer = w
                currentFile = file
                pendingSinceFlush = 0
                Log.d(TAG, "MetricsLogger.init: writing to ${file.absolutePath}")
            } catch (e: Exception) {
                Log.e(TAG, "MetricsLogger.init failed: ${e.message}", e)
                writer = null
                currentFile = null
            }
        }
    }

    fun log(record: MetricRecord) {
        synchronized(lock) {
            val w = writer ?: return@synchronized
            try {
                w.write(formatRow(record))
                w.newLine()
                pendingSinceFlush++
                if (pendingSinceFlush >= FLUSH_EVERY_N) {
                    w.flush()
                    pendingSinceFlush = 0
                }
            } catch (e: Exception) {
                Log.w(TAG, "MetricsLogger.log failed: ${e.message}")
            }
            Unit
        }
    }

    fun close() {
        synchronized(lock) {
            try {
                writer?.flush()
                writer?.close()
            } catch (e: Exception) {
                Log.w(TAG, "MetricsLogger.close failed: ${e.message}")
            }
            writer = null
            pendingSinceFlush = 0
        }
    }

    fun flush() {
        synchronized(lock) {
            try {
                writer?.flush()
                pendingSinceFlush = 0
            } catch (_: Exception) { /* ignore */ }
        }
    }

    fun getCurrentLogFile(): File? = synchronized(lock) { currentFile }

    fun getCurrentLogPath(): String =
        synchronized(lock) { currentFile?.absolutePath ?: "(no log)" }

    fun getCurrentLogName(): String =
        synchronized(lock) { currentFile?.name ?: "(no log)" }

    // -------------------------------------------------------------------------
    // Yardımcılar
    // -------------------------------------------------------------------------

    @Suppress("DEPRECATION")
    fun getCurrentBatteryPercent(context: Context): Int {
        return try {
            val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            val status = context.registerReceiver(null, filter) ?: return -1
            val level = status.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
            val scale = status.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
            if (level >= 0 && scale > 0) (level * 100) / scale else -1
        } catch (e: Exception) {
            Log.w(TAG, "battery read failed: ${e.message}")
            -1
        }
    }

    fun getNetworkType(context: Context): String {
        return try {
            val cm = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager
                ?: return "NONE"
            val net = cm.activeNetwork ?: return "NONE"
            val caps = cm.getNetworkCapabilities(net) ?: return "NONE"
            when {
                caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "WIFI"
                caps.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "MOBILE"
                caps.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ETHERNET"
                else -> "OTHER"
            }
        } catch (e: Exception) {
            Log.w(TAG, "network type read failed: ${e.message}")
            "NONE"
        }
    }

    private fun formatRow(r: MetricRecord): String {
        // Sayıları locale-bağımsız (US) yaz: tezde Excel/Pandas okuyacak.
        fun f(v: Float): String = String.format(Locale.US, "%.6f", v)
        return listOf(
            r.timestamp.toString(),
            csv(r.mode),
            csv(r.adaptiveDecision ?: ""),
            r.totalLatencyMs.toString(),
            r.melExtractionMs.toString(),
            r.inferenceMs.toString(),
            r.networkMs.toString(),
            r.rttMs?.toString() ?: "",
            r.bytesSent.toString(),
            r.bytesReceived.toString(),
            r.batteryPercent.toString(),
            csv(r.networkType),
            csv(r.prediction),
            csv(r.rawPrediction),
            f(r.confidence),
            r.entropy?.let { f(it) } ?: "",
            if (r.isSilence) "1" else "0",
            if (r.isUnknown) "1" else "0"
        ).joinToString(",")
    }

    private fun csv(v: String): String {
        // Basit CSV escape: virgül/quote/newline varsa quote'la
        return if (v.any { it == ',' || it == '"' || it == '\n' }) {
            "\"" + v.replace("\"", "\"\"") + "\""
        } else v
    }
}
