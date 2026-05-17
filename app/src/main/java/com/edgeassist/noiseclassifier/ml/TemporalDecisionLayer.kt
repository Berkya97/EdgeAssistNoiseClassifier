package com.edgeassist.noiseclassifier.ml

import android.util.Log
import com.edgeassist.noiseclassifier.data.model.DebugFrame
import com.edgeassist.noiseclassifier.data.model.InferenceResult
import kotlin.math.ln
import kotlin.math.max

/**
 * Zaman içinde karar stabilizasyonu (mel/olasılık tabanlı, RMS-variance boost yok):
 * - Rolling ortalama olasılık vektörü (son N frame).
 * - Entropy ve top1-top2 margin ile belirsizlik: yüksek entropy veya düşük margin → Unknown (K frame sürerse göster).
 * - Uyarlamalı confidence eşiği: base + k*entropy.
 * - Stickiness: önceki etiket hâlâ top1’e yakınsa flip etme.
 * - Minimum süre: etiket son N frame’de en az M kez kazanmış olmalı.
 * - Sessizlik: RMS ve ortalama mel enerjisi birlikte.
 *
 * Unknown oranını azaltmak için (MainViewModel'de): baseConfidenceThreshold ve
 * entropyCoeff düşür, entropyThresholdHigh yükselt, marginThresholdLow ve
 * minWinsToShow düşür, uncertainPersistenceFrames düşür. Tersi → daha muhafazakâr.
 */
class TemporalDecisionLayer(
    private val labels: List<String>,
    private val historySize: Int = 10,
    private val minWinsToShow: Int = 3,
    private val baseConfidenceThreshold: Float = 0.5f,
    private val entropyCoeff: Float = 0.25f,
    private val entropyThresholdHigh: Float = 2.0f,
    private val marginThresholdLow: Float = 0.06f,
    private val uncertainPersistenceFrames: Int = 4,
    private val stickinessDelta: Float = 0.05f,
    private val silenceThresholdRms: Float = 0.0003f,
    private val silenceThresholdMel: Float = -65f,
    private val enableDebugLogs: Boolean = false
) {

    companion object {
        private const val TAG = "NoiseClassifier"
        const val LABEL_UNKNOWN = "Unknown"
        const val LABEL_SILENCE = "Silence"
    }

    private val probHistory = ArrayDeque<FloatArray>(historySize)
    private var previousLabel: String? = null
    private var uncertainFrameCount: Int = 0
    private val winCounts = mutableMapOf<Int, Int>()

    /**
     * Ham model çıktısı + RMS + ortalama mel enerjisi (dB) ile smoothed sonuç ve debug frame.
     */
    fun process(
        rawResult: InferenceResult?,
        rms: Float,
        meanMelEnergy: Float = -80f
    ): Pair<InferenceResult, DebugFrame?> {
        val isSilence = rms < silenceThresholdRms || meanMelEnergy < silenceThresholdMel

        if (rawResult == null || isSilence) {
            val uniformProbs = FloatArray(labels.size) { 1f / labels.size }
            val silenceResult = InferenceResult(
                label = LABEL_SILENCE,
                confidence = 0f,
                inferenceTimeMs = 0,
                totalLatencyMs = rawResult?.totalLatencyMs ?: 0,
                probabilities = uniformProbs
            )
            return silenceResult to DebugFrame(
                rms = rms,
                meanMelEnergy = meanMelEnergy,
                entropy = 0f,
                top1Top2Margin = 0f,
                top1Prob = 0f,
                adaptiveThreshold = 0f,
                top3Labels = emptyList(),
                rollingMeanTop3 = emptyList(),
                rawLabel = LABEL_SILENCE,
                rawConfidence = 0f,
                smoothedLabel = LABEL_SILENCE,
                smoothedConfidence = 0f,
                unknownGatingReason = if (rawResult == null) "no_result" else "silence_rms_or_mel"
            )
        }

        val probs = rawResult.probabilities
        val numClasses = labels.size

        // Incoming probabilities logging: min/max/sum, first 10, rawConfidence vs probabilities[maxIndex]
        if (probs != null && probs.isNotEmpty()) {
            val pMin = probs.minOrNull() ?: 0f
            val pMax = probs.maxOrNull() ?: 0f
            val pSum = probs.sum()
            val first10 = probs.take(10).joinToString(", ") { "%.4f".format(it) }
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val probAtMax = probs[maxIdx]
            val rawConf = rawResult.confidence
            val match = kotlin.math.abs(probAtMax - rawConf) < 1e-5f
            Log.d(TAG, "[Temporal probs IN] min=%.4f max=%.4f sum=%.4f first10=[%s]".format(pMin, pMax, pSum, first10))
            Log.d(TAG, "[Temporal probs IN] rawConfidence=%.4f probabilities[maxIdx=%d]=%.4f match=%s (uniform check: max≈%.2f and sum≈1 → uniform)".format(rawConf, maxIdx, probAtMax, match, 1f / numClasses))
        } else {
            Log.d(TAG, "[Temporal probs IN] probabilities is null or empty; rawConfidence=%.4f".format(rawResult.confidence))
        }

        val workingProbs = if (probs != null && probs.size == numClasses) {
            probs.copyOf()
        } else {
            if (probHistory.isNotEmpty()) {
                rollingMean(probHistory)
            } else {
                FloatArray(numClasses) { 1f / numClasses }
            }
        }

        if (probs != null && probs.size == numClasses) {
            while (probHistory.size >= historySize) {
                val removed = probHistory.removeFirst()
                val oldWinner = removed.indices.maxByOrNull { removed[it] } ?: 0
                winCounts[oldWinner] = (winCounts[oldWinner] ?: 1) - 1
                if ((winCounts[oldWinner] ?: 0) <= 0) winCounts.remove(oldWinner)
            }
            probHistory.addLast(probs)
            val newWinner = probs.indices.maxByOrNull { probs[it] } ?: 0
            winCounts[newWinner] = (winCounts[newWinner] ?: 0) + 1
        }

        val rollingMeanProbs = when {
            probHistory.isNotEmpty() -> rollingMean(probHistory)
            workingProbs.size == numClasses -> workingProbs
            else -> FloatArray(numClasses) { 1f / numClasses }
        }
        if (rollingMeanProbs.size != numClasses) {
            return rawResult to null
        }

        // Array used for entropy (before computeEntropy): first 10 values
        val rollingFirst10 = rollingMeanProbs.take(10).joinToString(", ") { "%.4f".format(it) }
        Log.d(TAG, "[Temporal entropy input] rollingMeanProbs first10=[%s] (H≈2.303 if uniform)".format(rollingFirst10))

        val entropy = computeEntropy(rollingMeanProbs)
        val (top1Idx, top2Idx) = rollingMeanProbs.indices
            .sortedByDescending { rollingMeanProbs[it] }
            .take(2)
            .let { list -> Pair(list.getOrNull(0) ?: 0, list.getOrNull(1) ?: 0) }
        val top1Prob = rollingMeanProbs[top1Idx]
        val top2Prob = rollingMeanProbs[top2Idx]
        val margin = top1Prob - top2Prob

        val adaptiveThreshold = baseConfidenceThreshold + entropyCoeff * entropy
        val isUncertain = entropy >= entropyThresholdHigh || margin < marginThresholdLow

        if (isUncertain) {
            uncertainFrameCount++
        } else {
            uncertainFrameCount = 0
        }

        val showDespiteUncertain = uncertainFrameCount >= uncertainPersistenceFrames
        val useUnknown = isUncertain && !showDespiteUncertain

        val winsForTop1 = winCounts.getOrDefault(top1Idx, 0)
        val meetsMinDuration = winsForTop1 >= minWinsToShow

        var outputLabel: String
        var outputConf: Float
        var stickinessActive = false
        var unknownGatingReason: String? = null

        when {
            useUnknown -> {
                outputLabel = LABEL_UNKNOWN
                outputConf = top1Prob
                unknownGatingReason = when {
                    entropy >= entropyThresholdHigh -> "high_entropy"
                    margin < marginThresholdLow -> "low_margin"
                    else -> "uncertain"
                }
            }
            top1Prob < adaptiveThreshold -> {
                outputLabel = LABEL_UNKNOWN
                outputConf = top1Prob
                unknownGatingReason = "below_adaptive_threshold(%.2f)".format(adaptiveThreshold)
            }
            !meetsMinDuration -> {
                outputLabel = LABEL_UNKNOWN
                outputConf = top1Prob
                unknownGatingReason = "min_duration(wins=$winsForTop1,need=$minWinsToShow)"
            }
            else -> {
                val candidateLabel = labels.getOrElse(top1Idx) { LABEL_UNKNOWN }
                val prevLabel = previousLabel
                val prevIdx = labels.indexOf(prevLabel)
                val prevProb = if (prevIdx >= 0) rollingMeanProbs[prevIdx] else 0f

                if (prevLabel != null && prevLabel == candidateLabel) {
                    outputLabel = candidateLabel
                    outputConf = top1Prob
                } else if (prevLabel != null && prevIdx >= 0 && prevProb >= top1Prob - stickinessDelta && prevProb >= marginThresholdLow) {
                    outputLabel = prevLabel
                    outputConf = prevProb
                    stickinessActive = true
                } else {
                    outputLabel = candidateLabel
                    outputConf = top1Prob
                }
                previousLabel = outputLabel
            }
        }

        val rollingMeanTop3 = rollingMeanProbs.indices
            .map { i -> labels.getOrElse(i) { "?" } to rollingMeanProbs[i] }
            .sortedByDescending { it.second }
            .take(3)

        val top3 = if (workingProbs.size == numClasses) {
            workingProbs.indices
                .map { i -> labels.getOrElse(i) { "?" } to workingProbs[i] }
                .sortedByDescending { it.second }
                .take(3)
        } else {
            rollingMeanTop3
        }

        val finalResult = rawResult.copy(
            label = outputLabel,
            confidence = outputConf,
            probabilities = rollingMeanProbs
        )

        val debugFrame = DebugFrame(
            rms = rms,
            meanMelEnergy = meanMelEnergy,
            entropy = entropy,
            top1Top2Margin = margin,
            top1Prob = top1Prob,
            adaptiveThreshold = adaptiveThreshold,
            top3Labels = top3,
            rollingMeanTop3 = rollingMeanTop3,
            rawLabel = rawResult.label,
            rawConfidence = rawResult.confidence,
            smoothedLabel = outputLabel,
            smoothedConfidence = outputConf,
            stickinessActive = stickinessActive,
            unknownGatingReason = unknownGatingReason
        )

        if (enableDebugLogs) {
            Log.d(TAG, "Temporal: H=%.3f margin=%.3f thresh=%.3f raw=%s smooth=%s sticky=$stickinessActive reason=$unknownGatingReason"
                .format(entropy, margin, adaptiveThreshold, rawResult.label, outputLabel))
            Log.d(TAG, "  RollingTop3: ${rollingMeanTop3.joinToString { "${it.first}=%.3f".format(it.second) }}")
        }

        return finalResult to debugFrame
    }

    private fun computeEntropy(probs: FloatArray): Float {
        var h = 0f
        for (p in probs) {
            if (p > 1e-10f) h -= p * ln(p).toFloat()
        }
        return h
    }

    /**
     * Exponentially weighted mean: son frame'lere daha fazla ağırlık verir.
     * decay=0.6 → ağırlıklar (eskiden yeniye): 0.6^(n-1), 0.6^(n-2), ..., 0.6^0
     * Gun shot gibi kısa impulsif sesler için kritik: en son frame baskın olur.
     */
    private fun rollingMean(history: ArrayDeque<FloatArray>): FloatArray {
        if (history.isEmpty()) return FloatArray(0)
        val dim = history.first().size
        val decay = 0.6f
        val weighted = FloatArray(dim)
        var totalWeight = 0f
        var weight = 1f
        for (arr in history.reversed()) {
            for (i in 0 until minOf(dim, arr.size)) weighted[i] += arr[i] * weight
            totalWeight += weight
            weight *= decay
        }
        return FloatArray(dim) { weighted[it] / totalWeight }
    }

    fun reset() {
        probHistory.clear()
        previousLabel = null
        uncertainFrameCount = 0
        winCounts.clear()
    }
}
