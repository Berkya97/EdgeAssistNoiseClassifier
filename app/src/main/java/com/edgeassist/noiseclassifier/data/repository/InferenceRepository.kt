package com.edgeassist.noiseclassifier.data.repository

import android.util.Log
import com.edgeassist.noiseclassifier.adaptive.AdaptiveDecisionEngine
import com.edgeassist.noiseclassifier.data.model.ClassificationMode
import com.edgeassist.noiseclassifier.data.model.DebugFrame
import com.edgeassist.noiseclassifier.data.model.InferenceResult
import com.edgeassist.noiseclassifier.ml.MelSpectrogramExtractor
import com.edgeassist.noiseclassifier.ml.PredictionSmoother
import com.edgeassist.noiseclassifier.ml.TemporalDecisionLayer
import com.edgeassist.noiseclassifier.ml.TFLiteClassifier
import com.edgeassist.noiseclassifier.network.CloudApiClient

class InferenceRepository(
    private val melExtractor: MelSpectrogramExtractor,
    private val tfliteClassifier: TFLiteClassifier,
    private val cloudApiClient: CloudApiClient,
    private val adaptiveEngine: AdaptiveDecisionEngine,
    private val temporalLayer: TemporalDecisionLayer
) {

    companion object {
        private const val TAG = "NoiseClassifier"

        /** Mel spectrogram (dB) ortalaması; sessizlik kapısı için. */
        private fun computeMeanMelEnergyDb(melFeatures: Array<FloatArray>): Float {
            var sum = 0.0
            var count = 0
            for (row in melFeatures) for (v in row) {
                sum += v
                count++
            }
            return if (count == 0) -80f else (sum / count).toFloat()
        }
    }

    fun classify(mode: ClassificationMode, audioSamples: ShortArray): InferenceResult {
        val totalStart = System.nanoTime()

        val rms = PredictionSmoother.computeRms(audioSamples)
        val pcmMin = audioSamples.minOrNull()?.toInt() ?: 0
        val pcmMax = audioSamples.maxOrNull()?.toInt() ?: 0
        Log.d(TAG, "Raw PCM min=$pcmMin max=$pcmMax, RMS=%.6f, samples=${audioSamples.size}".format(rms))

        val melStart = System.nanoTime()
        val melFeatures = melExtractor.extract(audioSamples)
        val melTimeMs = (System.nanoTime() - melStart) / 1_000_000
        val meanMelEnergy = computeMeanMelEnergyDb(melFeatures)

        if (rms < 0.0003f || meanMelEnergy < -65f) {
            val totalMs = (System.nanoTime() - totalStart) / 1_000_000
            Log.d(TAG, "Silence gate: rms=%.5f mel=%.1f, skipping inference".format(rms, meanMelEnergy))
            val (silenceResult, debugFrame) = temporalLayer.process(null, rms, meanMelEnergy)
            return silenceResult.copy(
                totalLatencyMs = totalMs,
                melExtractionMs = melTimeMs,
                debugFrame = debugFrame
            )
        }
        Log.d(TAG, "Mel spectrogram extraction took ${melTimeMs}ms")

        val resolvedMode: ClassificationMode
        val result: InferenceResult

        when (mode) {
            ClassificationMode.EDGE -> {
                resolvedMode = ClassificationMode.EDGE
                result = runEdgeInference(melFeatures)
            }
            ClassificationMode.CLOUD -> {
                resolvedMode = ClassificationMode.CLOUD
                result = runCloudWithFallback(melFeatures)
            }
            ClassificationMode.ADAPTIVE -> {
                val (decidedMode, metrics) = adaptiveEngine.decide()
                resolvedMode = decidedMode
                Log.d(TAG, "Adaptive decided: $decidedMode (metrics=$metrics)")

                result = if (decidedMode == ClassificationMode.CLOUD) {
                    runCloudWithFallback(melFeatures)
                } else {
                    runEdgeInference(melFeatures)
                }
            }
        }

        val resultWithProbs = ensureProbabilities(result)
        val totalMs = (System.nanoTime() - totalStart) / 1_000_000

        if (resultWithProbs.probabilities == null || resultWithProbs.probabilities!!.isEmpty()) {
            val stack = Thread.currentThread().stackTrace
            val hint = stack.getOrNull(2)?.let { "${it.className}.${it.methodName}(${it.fileName}:${it.lineNumber})" } ?: "unknown"
            Log.e(TAG, "probabilities LOST before temporal: result.label=${resultWithProbs.label} probs=${resultWithProbs.probabilities?.size ?: "null"}, caller hint=$hint")
            Log.e(TAG, "  stack: ${stack.take(8).joinToString(" <- ") { "${it.methodName}:${it.lineNumber}" }}")
        }
        check(resultWithProbs.probabilities != null && resultWithProbs.probabilities!!.isNotEmpty()) {
            "probabilities must be non-null and non-empty before temporalLayer.process()"
        }

        val (finalFromTemporal, debugFrame) = temporalLayer.process(resultWithProbs, rms, meanMelEnergy)
        val finalResult = finalFromTemporal.copy(
            totalLatencyMs = totalMs,
            melExtractionMs = melTimeMs,
            inferenceTimeMs = resultWithProbs.inferenceTimeMs,
            networkMs = resultWithProbs.networkMs,
            bytesSent = resultWithProbs.bytesSent,
            bytesReceived = resultWithProbs.bytesReceived,
            modeUsed = resolvedMode,
            debugFrame = debugFrame
        )

        Log.d(TAG, "Classification complete: label=${finalResult.label}, " +
                "confidence=%.3f, ".format(finalResult.confidence) +
                "totalMs=$totalMs, mode=$resolvedMode")

        return finalResult
    }

    private fun runEdgeInference(melFeatures: Array<FloatArray>): InferenceResult {
        return tfliteClassifier.classify(melFeatures)
    }

    /** Ensures result has non-null, non-empty probabilities (zorunlu: Temporal null görmez). */
    private fun ensureProbabilities(result: InferenceResult): InferenceResult {
        val labels = tfliteClassifier.getLabels()
        val n = labels.size.coerceAtLeast(10)
        val existing = result.probabilities
        if (existing != null && existing.isNotEmpty() && existing.size == n) return result
        val idx = if (labels.isNotEmpty()) labels.indexOf(result.label).coerceIn(0, labels.size - 1) else 0
        val conf = result.confidence
        val probs = FloatArray(n) { i -> if (i == idx) conf else (1f - conf) / (n - 1).coerceAtLeast(1) }
        Log.d(TAG, "ensureProbabilities: filled synthetic size=$n (labels.size=${labels.size}) label=${result.label}")
        return result.copy(probabilities = probs)
    }

    fun resetTemporalLayer() {
        temporalLayer.reset()
    }

    private fun runCloudWithFallback(melFeatures: Array<FloatArray>): InferenceResult {
        return try {
            cloudApiClient.predict(melFeatures)
        } catch (e: Exception) {
            Log.w(TAG, "Cloud inference failed, falling back to EDGE: ${e.message}")
            val edgeResult = runEdgeInference(melFeatures)
            edgeResult.copy(modeUsed = ClassificationMode.EDGE)
        }
    }
}
