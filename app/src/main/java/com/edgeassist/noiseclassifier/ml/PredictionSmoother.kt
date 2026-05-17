package com.edgeassist.noiseclassifier.ml

import android.util.Log
import com.edgeassist.noiseclassifier.data.model.InferenceResult

/**
 * Gerçek zamanlı tahminleri stabilize eder:
 * - Confidence threshold: altında "Unknown" döner
 * - Majority vote: son N tahmin üzerinden en sık etiket
 * - Silence gate: düşük enerji için inference atlanır
 */
class PredictionSmoother(
    private val confidenceThreshold: Float = 0.0f,  // TemporalDecisionLayer zaten entropy-based thresholding yapıyor, double gating olmasın
    private val majorityVoteWindow: Int = 5,
    private val silenceThresholdRms: Float = 0.0003f,  // TemporalDecisionLayer ile aynı eşik
    private val enableDebugLogs: Boolean = false
) {

    companion object {
        private const val TAG = "NoiseClassifier"
        const val LABEL_UNKNOWN = "Unknown"
        const val LABEL_SILENCE = "Silence"

        /** PCM 16-bit ShortArray için RMS hesapla ([-1,1] float eşdeğeri) */
        fun computeRms(samples: ShortArray): Float {
            if (samples.isEmpty()) return 0f
            val scale = 1f / 32768f
            var sum = 0.0
            for (s in samples) {
                val x = s * scale
                sum += x * x
            }
            return kotlin.math.sqrt((sum / samples.size).toFloat())
        }
    }

    private val predictionHistory = ArrayDeque<Pair<String, Float>>(majorityVoteWindow)

    /**
     * Ham inference sonucunu smoothing uygulayarak döner.
     * Eğer rawResult null ise (örn. silence gate) LABEL_SILENCE döner.
     */
    fun smooth(rawResult: InferenceResult?, audioRms: Float? = null): InferenceResult {
        if (rawResult == null) {
            return InferenceResult(
                label = LABEL_SILENCE,
                confidence = 0f,
                inferenceTimeMs = 0,
                totalLatencyMs = 0
            )
        }

        if (audioRms != null && audioRms < silenceThresholdRms) {
            if (enableDebugLogs) Log.d(TAG, "Silence gate: RMS=%.6f < %.6f".format(audioRms, silenceThresholdRms))
            return InferenceResult(
                label = LABEL_SILENCE,
                confidence = 0f,
                inferenceTimeMs = rawResult.inferenceTimeMs,
                totalLatencyMs = rawResult.totalLatencyMs
            )
        }

        val label = if (rawResult.confidence < confidenceThreshold) {
            LABEL_UNKNOWN
        } else {
            rawResult.label
        }

        predictionHistory.addLast(label to rawResult.confidence)
        while (predictionHistory.size > majorityVoteWindow) {
            predictionHistory.removeFirst()
        }

        val votedLabel = majorityVote()
        val votedConfidence = rawResult.confidence

        if (enableDebugLogs && predictionHistory.size >= majorityVoteWindow) {
            Log.d(TAG, "Majority vote: raw=$label -> voted=$votedLabel (window=${predictionHistory.size})")
        }

        return rawResult.copy(
            label = votedLabel,
            confidence = votedConfidence
        )
    }

    private fun majorityVote(): String {
        if (predictionHistory.isEmpty()) return LABEL_UNKNOWN

        val counts = mutableMapOf<String, Int>()
        for ((l, _) in predictionHistory) {
            counts[l] = (counts[l] ?: 0) + 1
        }

        var maxCount = 0
        var winner = LABEL_UNKNOWN
        for ((label, count) in counts) {
            if (count > maxCount) {
                maxCount = count
                winner = label
            }
        }
        return winner
    }

    fun reset() {
        predictionHistory.clear()
    }
}
