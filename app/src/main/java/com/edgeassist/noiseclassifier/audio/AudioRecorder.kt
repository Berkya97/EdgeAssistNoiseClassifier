package com.edgeassist.noiseclassifier.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.isActive
import kotlin.coroutines.coroutineContext

/**
 * Ses pencereleri üretir.
 * - Overlapping: window=1s, stride=configurable (örn. 0.25s) → her stride'da son 1s emit edilir.
 * - Non-overlapping: stride=1s (varsayılan eski davranış).
 * MIC / UNPROCESSED (API 24+) desteklenir.
 */
class AudioRecorder(
    /** Pencere uzunluğu (saniye). Her emit tam bu kadar örnek içerir. */
    private val windowSeconds: Float = 1f,
    /** Stride (saniye). Bu kadar yeni örnek geldikten sonra bir sonraki pencere emit edilir. */
    private val strideSeconds: Float = 0.25f
) {

    companion object {
        private const val TAG = "NoiseClassifier"
        const val SAMPLE_RATE = 16000
        private const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        private const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    }

    val samplesPerWindow: Int = (SAMPLE_RATE * windowSeconds).toInt()
    private val strideSamples: Int = maxOf(1, (SAMPLE_RATE * strideSeconds).toInt())

    private var audioRecord: AudioRecord? = null

    @SuppressLint("MissingPermission")
    fun audioFlow(): Flow<ShortArray> = flow {
        val bufferSize = maxOf(
            AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT),
            samplesPerWindow * 2
        )

        var audioSource = MediaRecorder.AudioSource.MIC
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
            audioSource = MediaRecorder.AudioSource.UNPROCESSED
        }

        var recorder = AudioRecord(
            audioSource,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            bufferSize
        )

        if (recorder.state != AudioRecord.STATE_INITIALIZED) {
            Log.w(TAG, "UNPROCESSED failed, falling back to MIC")
            recorder.release()
            recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANNEL_CONFIG,
                AUDIO_FORMAT,
                bufferSize
            )
            if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord failed to initialize")
                recorder.release()
                return@flow
            }
        }

        audioRecord = recorder
        recorder.startRecording()
        Log.d(TAG, "Audio recording: ${SAMPLE_RATE}Hz mono, window=${samplesPerWindow} samples, stride=${strideSamples} samples")

        try {
            val ringBuffer = ShortArray(samplesPerWindow)
            var writeIndex = 0
            var samplesSinceEmit = 0
            val readChunkSize = 1024
            val readBuffer = ShortArray(readChunkSize)

            while (coroutineContext.isActive) {
                val read = recorder.read(readBuffer, 0, readChunkSize)
                if (read > 0) {
                    var srcOffset = 0
                    while (srcOffset < read) {
                        val toCopy = minOf(read - srcOffset, samplesPerWindow)
                        for (i in 0 until toCopy) {
                            ringBuffer[writeIndex] = readBuffer[srcOffset + i]
                            writeIndex = (writeIndex + 1) % samplesPerWindow
                        }
                        srcOffset += toCopy
                        samplesSinceEmit += toCopy

                        while (samplesSinceEmit >= strideSamples) {
                            val window = ShortArray(samplesPerWindow)
                            for (i in 0 until samplesPerWindow) {
                                window[i] = ringBuffer[(writeIndex + i) % samplesPerWindow]
                            }
                            emit(window)
                            samplesSinceEmit -= strideSamples
                        }
                    }
                } else if (read < 0) {
                    Log.e(TAG, "AudioRecord read error: $read")
                    break
                }
            }
        } finally {
            recorder.stop()
            recorder.release()
            audioRecord = null
            Log.d(TAG, "Audio recording stopped and released")
        }
    }.flowOn(Dispatchers.IO)

    fun stop() {
        try {
            audioRecord?.stop()
        } catch (e: Exception) {
            Log.w(TAG, "Error stopping AudioRecord: ${e.message}")
        }
    }
}
