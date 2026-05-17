package com.edgeassist.noiseclassifier.metrics

import android.content.Context
import android.util.Log
import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.Locale

/**
 * AKADEMİK ÖLÇÜM İÇİN KULLANIM KILAVUZU
 * =====================================
 * Güvenilir batarya delta ölçümü için:
 * 1. Telefonu şarjdan çıkar, en az 5 dk bekle (batarya stabilize olsun)
 * 2. Ekran parlaklığını sabit tut (örn %50)
 * 3. WiFi/Bluetooth/GPS durumunu her test için AYNI tut
 * 4. Uygulama dışındaki işlemleri kapat (Battery Saver KAPALI)
 * 5. Her modu EN AZ 5 DAKİKA çalıştır (Android batarya sayacı ~%1 çözünürlükte)
 *    - 5 dk edge: batarya ~%2-3 düşmeli
 *    - 5 dk cloud (WiFi): batarya ~%3-5 düşmeli
 * 6. Modlar arası soğuma süresi bırak (2-3 dk bekle)
 * 7. Her mod için 3 tekrar yap, ortalamasını al
 */

/**
 * Edge/Cloud/Adaptive modları arasında batarya etkisini karşılaştırmak için
 * oturum-seviyesi özet kaydedicisi.
 *
 * Singleton — her [startSession] yeni bir oturum başlatır, [endSession] biter.
 * CSV çıktısı TEK dosyada toplanır ve her oturum append edilir:
 *   <externalFilesDir>/metrics/session_summary.csv
 */
object SessionLogger {

    private const val TAG = "NoiseClassifier"

    data class SessionSummary(
        val sessionId: Long,
        val mode: String,
        val startTimestamp: Long,
        val endTimestamp: Long,
        val durationMs: Long,
        val startBatteryPercent: Int,
        val endBatteryPercent: Int,
        val batteryDelta: Int,
        val inferenceCount: Int,
        val silenceCount: Int,
        val avgTotalLatencyMs: Double,
        val avgInferenceMs: Double,
        val avgNetworkMs: Double,
        val totalBytesSent: Long,
        val totalBytesReceived: Long,
        val batteryDropPerMinute: Double
    ) {
        val silenceRatio: Double
            get() = if (inferenceCount > 0) silenceCount.toDouble() / inferenceCount else 0.0

        val durationSec: Double
            get() = durationMs / 1000.0
    }

    private class ActiveSession(
        val sessionId: Long,
        val mode: String,
        val startTimestamp: Long,
        val startBatteryPercent: Int
    ) {
        var inferenceCount: Int = 0
        var silenceCount: Int = 0
        var sumTotalLatencyMs: Long = 0L
        var sumInferenceMs: Long = 0L
        var sumNetworkMs: Long = 0L
        var totalBytesSent: Long = 0L
        var totalBytesReceived: Long = 0L
    }

    private val lock = Any()
    private var active: ActiveSession? = null

    private val CSV_HEADER = listOf(
        "session_id",
        "mode",
        "start_timestamp",
        "end_timestamp",
        "duration_sec",
        "start_battery",
        "end_battery",
        "battery_delta",
        "inference_count",
        "silence_count",
        "silence_ratio",
        "avg_total_latency_ms",
        "avg_inference_ms",
        "avg_network_ms",
        "total_bytes_sent",
        "total_bytes_received",
        "battery_drop_per_minute"
    )

    /** Yeni oturum başlatır. Mevcut oturum varsa önce onu bitirir (özet yazılır). */
    fun startSession(context: Context, mode: String): Long {
        synchronized(lock) {
            if (active != null) {
                endSessionLocked(context)
            }
            val sessionId = System.currentTimeMillis()
            val battery = MetricsLogger.getCurrentBatteryPercent(context)
            active = ActiveSession(
                sessionId = sessionId,
                mode = mode,
                startTimestamp = sessionId,
                startBatteryPercent = battery
            )
            Log.d(TAG, "SessionLogger.startSession: id=$sessionId mode=$mode battery=$battery%")
            return sessionId
        }
    }

    /** Oturumdaki her inference için çağrılır; toplamları biriktirir. */
    fun recordInference(
        totalMs: Long,
        inferenceMs: Long,
        networkMs: Long,
        bytesSent: Long,
        bytesReceived: Long,
        isSilence: Boolean
    ) {
        synchronized(lock) {
            val s = active ?: return
            s.inferenceCount++
            if (isSilence) s.silenceCount++
            s.sumTotalLatencyMs += totalMs
            s.sumInferenceMs += inferenceMs
            s.sumNetworkMs += networkMs
            s.totalBytesSent += bytesSent
            s.totalBytesReceived += bytesReceived
        }
    }

    /** Aktif oturumu bitirir, CSV'ye yazar ve özeti döndürür. */
    fun endSession(context: Context): SessionSummary? {
        synchronized(lock) {
            return endSessionLocked(context)
        }
    }

    private fun endSessionLocked(context: Context): SessionSummary? {
        val s = active ?: return null
        active = null

        val endTs = System.currentTimeMillis()
        val endBattery = MetricsLogger.getCurrentBatteryPercent(context)
        val durationMs = endTs - s.startTimestamp
        val delta = if (s.startBatteryPercent >= 0 && endBattery >= 0) {
            s.startBatteryPercent - endBattery
        } else 0

        val n = s.inferenceCount
        val avgTotal = if (n > 0) s.sumTotalLatencyMs.toDouble() / n else 0.0
        val avgInf = if (n > 0) s.sumInferenceMs.toDouble() / n else 0.0
        val avgNet = if (n > 0) s.sumNetworkMs.toDouble() / n else 0.0

        val dropPerMinute = if (durationMs > 0) {
            delta.toDouble() * 60_000.0 / durationMs.toDouble()
        } else 0.0

        val summary = SessionSummary(
            sessionId = s.sessionId,
            mode = s.mode,
            startTimestamp = s.startTimestamp,
            endTimestamp = endTs,
            durationMs = durationMs,
            startBatteryPercent = s.startBatteryPercent,
            endBatteryPercent = endBattery,
            batteryDelta = delta,
            inferenceCount = n,
            silenceCount = s.silenceCount,
            avgTotalLatencyMs = avgTotal,
            avgInferenceMs = avgInf,
            avgNetworkMs = avgNet,
            totalBytesSent = s.totalBytesSent,
            totalBytesReceived = s.totalBytesReceived,
            batteryDropPerMinute = dropPerMinute
        )

        appendToCsv(context, summary)
        Log.d(TAG, "SessionLogger.endSession: ${summary.mode} dur=${summary.durationSec}s " +
                "delta=${summary.batteryDelta}% n=${summary.inferenceCount}")
        return summary
    }

    /** Aktif oturumun canlı özeti (henüz bitmemiş — UI için). */
    fun getCurrentSummary(): SessionSummary? {
        synchronized(lock) {
            val s = active ?: return null
            val now = System.currentTimeMillis()
            val durationMs = now - s.startTimestamp
            val n = s.inferenceCount
            return SessionSummary(
                sessionId = s.sessionId,
                mode = s.mode,
                startTimestamp = s.startTimestamp,
                endTimestamp = now,
                durationMs = durationMs,
                startBatteryPercent = s.startBatteryPercent,
                endBatteryPercent = -1,
                batteryDelta = 0,
                inferenceCount = n,
                silenceCount = s.silenceCount,
                avgTotalLatencyMs = if (n > 0) s.sumTotalLatencyMs.toDouble() / n else 0.0,
                avgInferenceMs = if (n > 0) s.sumInferenceMs.toDouble() / n else 0.0,
                avgNetworkMs = if (n > 0) s.sumNetworkMs.toDouble() / n else 0.0,
                totalBytesSent = s.totalBytesSent,
                totalBytesReceived = s.totalBytesReceived,
                batteryDropPerMinute = 0.0
            )
        }
    }

    /**
     * Çok kaba tahmin: inference başına ortalama batarya düşüşü (%).
     * Yalnızca [inferenceCount] > 0 VE [batteryDelta] > 0 ise anlamlıdır.
     */
    fun estimateEnergyPerInference(summary: SessionSummary): Double {
        if (summary.inferenceCount <= 0 || summary.batteryDelta <= 0) return 0.0
        return summary.batteryDelta.toDouble() / summary.inferenceCount
    }

    private fun appendToCsv(context: Context, summary: SessionSummary) {
        try {
            val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
            val metricsDir = File(baseDir, "metrics").apply { mkdirs() }
            val file = File(metricsDir, "session_summary.csv")
            val writeHeader = !file.exists() || file.length() == 0L
            BufferedWriter(FileWriter(file, true)).use { w ->
                if (writeHeader) {
                    w.write(CSV_HEADER.joinToString(","))
                    w.newLine()
                }
                w.write(formatRow(summary))
                w.newLine()
            }
        } catch (e: Exception) {
            Log.w(TAG, "SessionLogger.appendToCsv failed: ${e.message}")
        }
    }

    fun getSummaryFile(context: Context): File {
        val baseDir = context.getExternalFilesDir(null) ?: context.filesDir
        return File(File(baseDir, "metrics"), "session_summary.csv")
    }

    private fun formatRow(s: SessionSummary): String {
        fun d(v: Double): String = String.format(Locale.US, "%.6f", v)
        return listOf(
            s.sessionId.toString(),
            s.mode,
            s.startTimestamp.toString(),
            s.endTimestamp.toString(),
            d(s.durationSec),
            s.startBatteryPercent.toString(),
            s.endBatteryPercent.toString(),
            s.batteryDelta.toString(),
            s.inferenceCount.toString(),
            s.silenceCount.toString(),
            d(s.silenceRatio),
            d(s.avgTotalLatencyMs),
            d(s.avgInferenceMs),
            d(s.avgNetworkMs),
            s.totalBytesSent.toString(),
            s.totalBytesReceived.toString(),
            d(s.batteryDropPerMinute)
        ).joinToString(",")
    }
}
