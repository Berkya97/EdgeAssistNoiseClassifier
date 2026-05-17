package com.edgeassist.noiseclassifier.network

import android.util.Log
import com.edgeassist.noiseclassifier.data.model.ClassificationMode
import com.edgeassist.noiseclassifier.data.model.InferenceResult
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException
import java.util.concurrent.TimeUnit

class CloudApiClient(
    private val baseUrl: String = DEFAULT_BASE_URL
) {

    companion object {
        private const val TAG = "NoiseClassifier"
        const val DEFAULT_BASE_URL = "http://192.168.1.107:8000"
        private const val CONNECT_TIMEOUT_SEC = 5L
        private const val READ_TIMEOUT_SEC = 5L
        private val JSON_MEDIA_TYPE = "application/json; charset=utf-8".toMediaType()
    }

    private val client = OkHttpClient.Builder()
        .connectTimeout(CONNECT_TIMEOUT_SEC, TimeUnit.SECONDS)
        .readTimeout(READ_TIMEOUT_SEC, TimeUnit.SECONDS)
        .build()

    private val gson = Gson()

    /** Mel spectrogram features (64 x T), shape matches training. Server must accept this format. */
    data class PredictRequest(val mfcc: Array<FloatArray>)

    data class PredictResponse(
        val label: String,
        val confidence: Float,
        @SerializedName("inference_ms") val inferenceMs: Long
    )

    fun predict(mfccFeatures: Array<FloatArray>): InferenceResult {
        val totalStart = System.nanoTime()

        val requestBody = gson.toJson(PredictRequest(mfccFeatures))
        val bytesSent = requestBody.toByteArray().size

        Log.d(TAG, "Cloud predict: sending $bytesSent bytes to $baseUrl/predict")

        val request = Request.Builder()
            .url("$baseUrl/predict")
            .post(requestBody.toRequestBody(JSON_MEDIA_TYPE))
            .build()

        try {
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) {
                    throw IOException("Server returned ${response.code}")
                }

                val body = response.body?.string()
                    ?: throw IOException("Empty response body")
                val bytesReceived = body.toByteArray().size

                val result = gson.fromJson(body, PredictResponse::class.java)
                val totalMs = (System.nanoTime() - totalStart) / 1_000_000

                Log.d(TAG, "Cloud predict: label=${result.label}, " +
                        "confidence=${result.confidence}, " +
                        "serverInference=${result.inferenceMs}ms, " +
                        "totalLatency=${totalMs}ms, bytesRx=$bytesReceived")

                return InferenceResult(
                    label = result.label,
                    confidence = result.confidence,
                    inferenceTimeMs = result.inferenceMs,
                    totalLatencyMs = totalMs,
                    bytesSent = bytesSent,
                    bytesReceived = bytesReceived,
                    networkMs = totalMs,
                    modeUsed = ClassificationMode.CLOUD
                )
            }
        } catch (e: Exception) {
            val totalMs = (System.nanoTime() - totalStart) / 1_000_000
            Log.e(TAG, "Cloud predict failed after ${totalMs}ms: ${e.message}")
            throw e
        }
    }

    /**
     * Measure round-trip time to the server via GET /ping.
     * Returns RTT in milliseconds, or Long.MAX_VALUE on failure.
     */
    fun ping(): Long {
        val start = System.nanoTime()
        return try {
            val request = Request.Builder()
                .url("$baseUrl/ping")
                .get()
                .build()

            client.newCall(request).execute().use { response ->
                val rtt = (System.nanoTime() - start) / 1_000_000
                if (response.isSuccessful) {
                    Log.d(TAG, "Ping RTT: ${rtt}ms")
                    rtt
                } else {
                    Log.w(TAG, "Ping failed with status ${response.code}")
                    Long.MAX_VALUE
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Ping failed: ${e.message}")
            Long.MAX_VALUE
        }
    }
}
