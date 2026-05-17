package com.edgeassist.noiseclassifier.adaptive

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.util.Log
import com.edgeassist.noiseclassifier.data.model.ClassificationMode
import com.edgeassist.noiseclassifier.data.model.Metrics
import com.edgeassist.noiseclassifier.network.CloudApiClient
import java.io.RandomAccessFile

class AdaptiveDecisionEngine(
    private val context: Context,
    private val cloudApiClient: CloudApiClient
) {

    companion object {
        private const val TAG = "NoiseClassifier"
        private const val RTT_THRESHOLD_MS = 150L
        private const val BATTERY_THRESHOLD_PERCENT = 25
        private const val CPU_THRESHOLD_PERCENT = 80f

        // Cache TTL: ölçümleri her inference'da yeniden yapmayalım;
        // Thread.sleep(100) + ping() audio flow'u bloke edip silence'a düşürüyor.
        private const val RTT_CACHE_TTL_MS = 2000L
        private const val CPU_CACHE_TTL_MS = 2000L
        private const val BATTERY_CACHE_TTL_MS = 5000L
    }

    @Volatile private var cachedRtt: Long = Long.MAX_VALUE
    @Volatile private var rttTimestamp: Long = 0L
    @Volatile private var cachedCpu: Float = 0f
    @Volatile private var cpuTimestamp: Long = 0L
    @Volatile private var cachedBattery: Int = 100
    @Volatile private var batteryTimestamp: Long = 0L

    fun decide(): Pair<ClassificationMode, Metrics> {
        val rtt = getRttCached()
        val battery = getBatteryCached()
        val cpu = getCpuCached()

        val metrics = Metrics(rttMs = rtt, batteryLevel = battery, cpuLoad = cpu)

        val mode = when {
            rtt > RTT_THRESHOLD_MS -> {
                Log.d(TAG, "Adaptive -> EDGE: RTT ${rtt}ms > ${RTT_THRESHOLD_MS}ms")
                ClassificationMode.EDGE
            }
            battery < BATTERY_THRESHOLD_PERCENT -> {
                Log.d(TAG, "Adaptive -> EDGE: Battery $battery% < $BATTERY_THRESHOLD_PERCENT%")
                ClassificationMode.EDGE
            }
            cpu > CPU_THRESHOLD_PERCENT -> {
                val cpuStr = "%.1f".format(cpu)
                Log.d(TAG, "Adaptive -> CLOUD: CPU load $cpuStr% > $CPU_THRESHOLD_PERCENT%")
                ClassificationMode.CLOUD
            }
            else -> {
                val cpuStr = "%.1f".format(cpu)
                Log.d(TAG, "Adaptive -> CLOUD: All conditions normal (RTT=${rtt}ms, bat=$battery%, cpu=$cpuStr%)")
                ClassificationMode.CLOUD
            }
        }

        Log.d(TAG, "Adaptive decision: $mode | metrics=$metrics")
        return mode to metrics
    }

    private fun getRttCached(): Long {
        val now = System.currentTimeMillis()
        if (now - rttTimestamp < RTT_CACHE_TTL_MS) return cachedRtt
        cachedRtt = measureRtt()
        rttTimestamp = now
        return cachedRtt
    }

    private fun getCpuCached(): Float {
        val now = System.currentTimeMillis()
        if (now - cpuTimestamp < CPU_CACHE_TTL_MS) return cachedCpu
        cachedCpu = getCpuLoad()
        cpuTimestamp = now
        return cachedCpu
    }

    private fun getBatteryCached(): Int {
        val now = System.currentTimeMillis()
        if (now - batteryTimestamp < BATTERY_CACHE_TTL_MS) return cachedBattery
        cachedBattery = getBatteryLevel()
        batteryTimestamp = now
        return cachedBattery
    }

    private fun measureRtt(): Long {
        return try {
            cloudApiClient.ping()
        } catch (e: Exception) {
            Log.w(TAG, "RTT measurement failed: ${e.message}")
            Long.MAX_VALUE
        }
    }

    @Suppress("DEPRECATION")
    private fun getBatteryLevel(): Int {
        return try {
            val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
            val batteryStatus = context.registerReceiver(null, filter)
            if (batteryStatus != null) {
                val level = batteryStatus.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = batteryStatus.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                if (level >= 0 && scale > 0) {
                    (level * 100) / scale
                } else {
                    100
                }
            } else {
                100
            }
        } catch (e: Exception) {
            Log.w(TAG, "Battery level read failed: ${e.message}")
            100
        }
    }

    private fun getCpuLoad(): Float {
        return try {
            val reader = RandomAccessFile("/proc/stat", "r")
            val firstLine = reader.readLine()
            reader.close()

            val parts = firstLine.split("\\s+".toRegex())
            if (parts.size < 5) return 0f

            val user1 = parts[1].toLong()
            val nice1 = parts[2].toLong()
            val system1 = parts[3].toLong()
            val idle1 = parts[4].toLong()
            val total1 = user1 + nice1 + system1 + idle1

            Thread.sleep(100)

            val reader2 = RandomAccessFile("/proc/stat", "r")
            val secondLine = reader2.readLine()
            reader2.close()

            val parts2 = secondLine.split("\\s+".toRegex())
            if (parts2.size < 5) return 0f

            val user2 = parts2[1].toLong()
            val nice2 = parts2[2].toLong()
            val system2 = parts2[3].toLong()
            val idle2 = parts2[4].toLong()
            val total2 = user2 + nice2 + system2 + idle2

            val totalDiff = total2 - total1
            val idleDiff = idle2 - idle1

            if (totalDiff <= 0) return 0f
            ((totalDiff - idleDiff).toFloat() / totalDiff) * 100f
        } catch (e: Exception) {
            Log.w(TAG, "CPU load estimation failed: ${e.message}")
            0f
        }
    }
}
