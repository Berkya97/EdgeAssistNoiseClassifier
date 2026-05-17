package com.edgeassist.noiseclassifier.data.model

data class InferenceResult(
    val label: String,
    val confidence: Float,
    val inferenceTimeMs: Long,
    val totalLatencyMs: Long,
    val bytesSent: Int = 0,
    val bytesReceived: Int = 0,
    val batteryDelta: Float = 0f,
    val modeUsed: ClassificationMode = ClassificationMode.EDGE,
    /** Mel spectrogram çıkarım süresi (ms). */
    val melExtractionMs: Long = 0,
    /** Cloud için toplam HTTP latency (ms); edge için 0. */
    val networkMs: Long = 0,
    /** Tüm sınıf olasılıkları (temporal smoothing / debug için) */
    val probabilities: FloatArray? = null,
    /** Son frame debug (panel için) */
    val debugFrame: DebugFrame? = null
) {
    override fun equals(other: Any?): Boolean = other is InferenceResult &&
        label == other.label && confidence == other.confidence &&
        inferenceTimeMs == other.inferenceTimeMs && totalLatencyMs == other.totalLatencyMs &&
        bytesSent == other.bytesSent && bytesReceived == other.bytesReceived &&
        batteryDelta == other.batteryDelta && modeUsed == other.modeUsed &&
        melExtractionMs == other.melExtractionMs && networkMs == other.networkMs &&
        probabilities.contentEquals(other.probabilities) &&
        debugFrame == other.debugFrame

    override fun hashCode(): Int {
        var result = label.hashCode()
        result = 31 * result + confidence.hashCode()
        result = 31 * result + inferenceTimeMs.hashCode()
        result = 31 * result + totalLatencyMs.hashCode()
        result = 31 * result + bytesSent
        result = 31 * result + bytesReceived
        result = 31 * result + batteryDelta.hashCode()
        result = 31 * result + modeUsed.hashCode()
        result = 31 * result + melExtractionMs.hashCode()
        result = 31 * result + networkMs.hashCode()
        result = 31 * result + (probabilities?.contentHashCode() ?: 0)
        result = 31 * result + (debugFrame?.hashCode() ?: 0)
        return result
    }
}
