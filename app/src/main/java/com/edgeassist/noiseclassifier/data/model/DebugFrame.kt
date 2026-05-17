package com.edgeassist.noiseclassifier.data.model

/**
 * Her frame için debug bilgisi: entropy, margin, rolling mean top3, stickiness, unknown gating nedeni.
 */
data class DebugFrame(
    val rms: Float,
    val meanMelEnergy: Float,
    val entropy: Float,
    val top1Top2Margin: Float,
    val top1Prob: Float = 0f,
    val adaptiveThreshold: Float = 0f,
    val top3Labels: List<Pair<String, Float>>,
    val rollingMeanTop3: List<Pair<String, Float>>,
    val rawLabel: String,
    val rawConfidence: Float,
    val smoothedLabel: String,
    val smoothedConfidence: Float,
    val stickinessActive: Boolean = false,
    val unknownGatingReason: String? = null,
    @Deprecated("Replaced by entropy/margin-based logic")
    val energyVariance: Float = 0f,
    @Deprecated("No longer used")
    val impulsiveBoostActive: Boolean = false,
    @Deprecated("No longer used")
    val steadyBoostActive: Boolean = false
)
