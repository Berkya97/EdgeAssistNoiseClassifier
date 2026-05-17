package com.edgeassist.noiseclassifier.ui

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.edgeassist.noiseclassifier.adaptive.AdaptiveDecisionEngine
import com.edgeassist.noiseclassifier.audio.AudioRecorder
import com.edgeassist.noiseclassifier.data.model.ClassificationMode
import com.edgeassist.noiseclassifier.data.model.InferenceResult
import com.edgeassist.noiseclassifier.data.repository.InferenceRepository
import com.edgeassist.noiseclassifier.metrics.MetricsLogger
import com.edgeassist.noiseclassifier.metrics.SessionLogger
import com.edgeassist.noiseclassifier.ml.MelSpectrogramExtractor
import com.edgeassist.noiseclassifier.ml.TFLiteClassifier
import com.edgeassist.noiseclassifier.ml.TemporalDecisionLayer
import com.edgeassist.noiseclassifier.ml.TemporalDecisionLayer.Companion.LABEL_SILENCE
import com.edgeassist.noiseclassifier.ml.TemporalDecisionLayer.Companion.LABEL_UNKNOWN
import com.edgeassist.noiseclassifier.network.CloudApiClient
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

class MainViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "NoiseClassifier"
        private const val DEBUG_LOGS = false
        private const val WINDOW_SECONDS = 1f
        private const val STRIDE_SECONDS = 0.25f
    }

    private val audioRecorder = AudioRecorder(
        windowSeconds = WINDOW_SECONDS,
        strideSeconds = STRIDE_SECONDS
    )
    private val melExtractor = MelSpectrogramExtractor(enableDebugLogs = DEBUG_LOGS)
    private val tfliteClassifier = TFLiteClassifier(application, enableDebugLogs = DEBUG_LOGS)
    private val cloudApiClient = CloudApiClient()
    private val adaptiveEngine = AdaptiveDecisionEngine(application, cloudApiClient)

    private lateinit var repository: InferenceRepository

    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private val _selectedMode = MutableStateFlow(ClassificationMode.EDGE)
    val selectedMode: StateFlow<ClassificationMode> = _selectedMode.asStateFlow()

    private val _latestResult = MutableStateFlow<InferenceResult?>(null)
    val latestResult: StateFlow<InferenceResult?> = _latestResult.asStateFlow()

    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    private val _inferenceCount = MutableStateFlow(0)
    val inferenceCount: StateFlow<Int> = _inferenceCount.asStateFlow()

    private val _logFileName = MutableStateFlow("")
    val logFileName: StateFlow<String> = _logFileName.asStateFlow()

    private val _lastSessionSummary = MutableStateFlow<SessionLogger.SessionSummary?>(null)
    val lastSessionSummary: StateFlow<SessionLogger.SessionSummary?> = _lastSessionSummary.asStateFlow()

    private var recordingJob: Job? = null

    init {
        MetricsLogger.init(application)
        _logFileName.value = MetricsLogger.getCurrentLogName()
        tfliteClassifier.initialize()
        val temporalLayer = TemporalDecisionLayer(
            labels = tfliteClassifier.getLabels(),
            historySize = 4,
            minWinsToShow = 1,
            baseConfidenceThreshold = 0.25f,
            entropyCoeff = 0.03f,
            entropyThresholdHigh = 2.25f,
            marginThresholdLow = 0.03f,
            uncertainPersistenceFrames = 2,
            stickinessDelta = 0.03f,
            silenceThresholdRms = 0.0003f,
            silenceThresholdMel = -65f,
            enableDebugLogs = DEBUG_LOGS
        )
        repository = InferenceRepository(
            melExtractor, tfliteClassifier, cloudApiClient, adaptiveEngine, temporalLayer
        )
    }

    fun setMode(mode: ClassificationMode) {
        val wasRecording = _isRecording.value
        val previousMode = _selectedMode.value
        _selectedMode.value = mode
        Log.d(TAG, "Mode changed to: $mode")

        // Kayıt devam ediyorsa: mevcut oturumu bitir, yeni modla yeni oturum başlat.
        if (wasRecording && previousMode != mode) {
            val ctx = getApplication<Application>().applicationContext
            val summary = SessionLogger.endSession(ctx)
            if (summary != null) _lastSessionSummary.value = summary
            SessionLogger.startSession(ctx, mode.name.lowercase())
        }
    }

    fun toggleRecording() {
        if (_isRecording.value) {
            stopRecording()
        } else {
            startRecording()
        }
    }

    private fun startRecording() {
        if (_isRecording.value) return

        _isRecording.value = true
        _inferenceCount.value = 0
        _lastSessionSummary.value = null
        repository.resetTemporalLayer()
        SessionLogger.startSession(
            getApplication<Application>().applicationContext,
            _selectedMode.value.name.lowercase()
        )
        Log.d(TAG, "Starting recording (overlapping stride=${STRIDE_SECONDS}s)...")

        recordingJob = viewModelScope.launch(Dispatchers.Default) {
            audioRecorder.audioFlow().collect { audioChunk ->
                if (!_isRecording.value) return@collect

                _isProcessing.value = true
                try {
                    val mode = _selectedMode.value
                    val result = repository.classify(mode, audioChunk)
                    _latestResult.value = result
                    _inferenceCount.value += 1

                    logMetricRecord(mode, result)

                    SessionLogger.recordInference(
                        totalMs = result.totalLatencyMs,
                        inferenceMs = result.inferenceTimeMs,
                        networkMs = result.networkMs,
                        bytesSent = result.bytesSent.toLong(),
                        bytesReceived = result.bytesReceived.toLong(),
                        isSilence = result.label == LABEL_SILENCE
                    )

                    Log.d(TAG, "Inference #${_inferenceCount.value}: ${result.label} (${(result.confidence * 100).toInt()}%) in ${result.totalLatencyMs}ms")
                } catch (e: Exception) {
                    Log.e(TAG, "Inference pipeline error: ${e.message}", e)
                } finally {
                    _isProcessing.value = false
                }
            }
        }
    }

    private fun stopRecording() {
        _isRecording.value = false
        recordingJob?.cancel()
        recordingJob = null
        audioRecorder.stop()
        repository.resetTemporalLayer()
        _isProcessing.value = false
        val summary = SessionLogger.endSession(getApplication<Application>().applicationContext)
        if (summary != null) _lastSessionSummary.value = summary
        Log.d(TAG, "Recording stopped. Total inferences: ${_inferenceCount.value}")
    }

    private fun logMetricRecord(selectedMode: ClassificationMode, result: InferenceResult) {
        try {
            val ctx = getApplication<Application>().applicationContext
            val debug = result.debugFrame
            val adaptiveDecision: String? = if (selectedMode == ClassificationMode.ADAPTIVE) {
                when (result.modeUsed) {
                    ClassificationMode.EDGE -> "edge"
                    ClassificationMode.CLOUD -> "cloud"
                    else -> null
                }
            } else null

            val rawPrediction = debug?.rawLabel ?: result.label
            val entropy = debug?.entropy
            val isSilence = result.label == LABEL_SILENCE
            val isUnknown = result.label == LABEL_UNKNOWN

            val record = MetricsLogger.MetricRecord(
                timestamp = System.currentTimeMillis(),
                mode = selectedMode.name.lowercase(),
                adaptiveDecision = adaptiveDecision,
                totalLatencyMs = result.totalLatencyMs,
                melExtractionMs = result.melExtractionMs,
                inferenceMs = result.inferenceTimeMs,
                networkMs = result.networkMs,
                rttMs = null,
                bytesSent = result.bytesSent.toLong(),
                bytesReceived = result.bytesReceived.toLong(),
                batteryPercent = MetricsLogger.getCurrentBatteryPercent(ctx),
                networkType = MetricsLogger.getNetworkType(ctx),
                prediction = result.label,
                rawPrediction = rawPrediction,
                confidence = result.confidence,
                entropy = entropy,
                isSilence = isSilence,
                isUnknown = isUnknown
            )
            MetricsLogger.log(record)
        } catch (e: Exception) {
            Log.w(TAG, "logMetricRecord failed: ${e.message}")
        }
    }

    override fun onCleared() {
        super.onCleared()
        stopRecording()
        tfliteClassifier.close()
        MetricsLogger.close()
        Log.d(TAG, "ViewModel cleared, resources released")
    }
}
