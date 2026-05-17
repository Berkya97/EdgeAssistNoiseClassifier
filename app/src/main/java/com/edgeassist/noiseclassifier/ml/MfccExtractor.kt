package com.edgeassist.noiseclassifier.ml

import android.util.Log
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

class MfccExtractor(
    val numMfcc: Int = DEFAULT_NUM_MFCC,
    val numFrames: Int = DEFAULT_NUM_FRAMES,
    private val sampleRate: Int = DEFAULT_SAMPLE_RATE,
    private val frameLength: Int = DEFAULT_FRAME_LENGTH,
    private val frameStep: Int = DEFAULT_FRAME_STEP,
    private val numMelFilters: Int = DEFAULT_NUM_MEL_FILTERS,
    private val nfft: Int = DEFAULT_NFFT,
    private val preEmphasis: Double = DEFAULT_PRE_EMPHASIS
) {

    companion object {
        private const val TAG = "NoiseClassifier"
        const val DEFAULT_NUM_MFCC = 13
        const val DEFAULT_NUM_FRAMES = 98
        const val DEFAULT_SAMPLE_RATE = 16000
        const val DEFAULT_FRAME_LENGTH = 400   // 25ms at 16kHz
        const val DEFAULT_FRAME_STEP = 160     // 10ms at 16kHz
        const val DEFAULT_NUM_MEL_FILTERS = 26
        const val DEFAULT_NFFT = 512
        const val DEFAULT_PRE_EMPHASIS = 0.97
    }

    private val hammingWindow = FloatArray(frameLength) { i ->
        (0.54 - 0.46 * cos(2.0 * PI * i / (frameLength - 1))).toFloat()
    }

    private val melFilterbank: Array<FloatArray> by lazy { buildMelFilterbank() }
    private val dctMatrix: Array<FloatArray> by lazy { buildDctMatrix() }

    fun extract(audioSamples: ShortArray): Array<FloatArray> {
        val startTime = System.nanoTime()

        val signal = applyPreEmphasis(audioSamples)
        val frames = frameSignal(signal)
        val actualFrames = frames.size

        val powerSpectra = Array(actualFrames) { i ->
            computePowerSpectrum(applyWindow(frames[i]))
        }

        val melEnergies = Array(actualFrames) { i ->
            applyMelFilterbank(powerSpectra[i])
        }

        val logMelEnergies = Array(actualFrames) { i ->
            FloatArray(numMelFilters) { j ->
                ln(melEnergies[i][j].toDouble().coerceAtLeast(1e-10)).toFloat()
            }
        }

        val mfccRaw = Array(actualFrames) { i ->
            applyDct(logMelEnergies[i])
        }

        val result = padOrTruncate(mfccRaw, numFrames, numMfcc)

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0
        Log.d(TAG, "MFCC extraction: ${audioSamples.size} samples -> " +
                "$actualFrames frames -> [$numFrames x $numMfcc] in %.2f ms".format(elapsedMs))

        return result
    }

    private fun applyPreEmphasis(samples: ShortArray): FloatArray {
        val signal = FloatArray(samples.size)
        signal[0] = samples[0].toFloat()
        for (i in 1 until samples.size) {
            signal[i] = samples[i] - (preEmphasis * samples[i - 1]).toFloat()
        }
        return signal
    }

    private fun frameSignal(signal: FloatArray): Array<FloatArray> {
        val numFramesActual = if (signal.size <= frameLength) {
            1
        } else {
            1 + (signal.size - frameLength) / frameStep
        }

        return Array(numFramesActual) { i ->
            val start = i * frameStep
            FloatArray(frameLength) { j ->
                val idx = start + j
                if (idx < signal.size) signal[idx] else 0f
            }
        }
    }

    private fun applyWindow(frame: FloatArray): FloatArray {
        return FloatArray(frameLength) { i -> frame[i] * hammingWindow[i] }
    }

    private fun computePowerSpectrum(windowedFrame: FloatArray): FloatArray {
        val real = FloatArray(nfft)
        val imag = FloatArray(nfft)
        for (i in windowedFrame.indices) {
            real[i] = windowedFrame[i]
        }

        fft(real, imag)

        val spectrumSize = nfft / 2 + 1
        return FloatArray(spectrumSize) { i ->
            (real[i] * real[i] + imag[i] * imag[i]) / nfft.toFloat()
        }
    }

    private fun fft(real: FloatArray, imag: FloatArray) {
        val n = real.size
        if (n <= 1) return

        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                var temp = real[i]; real[i] = real[j]; real[j] = temp
                temp = imag[i]; imag[i] = imag[j]; imag[j] = temp
            }
            var m = n / 2
            while (m >= 1 && j >= m) {
                j -= m
                m /= 2
            }
            j += m
        }

        var len = 2
        while (len <= n) {
            val halfLen = len / 2
            val angle = -2.0 * PI / len
            val wReal = cos(angle).toFloat()
            val wImag = sin(angle).toFloat()

            var i = 0
            while (i < n) {
                var curReal = 1.0f
                var curImag = 0.0f

                for (k in 0 until halfLen) {
                    val tReal = curReal * real[i + k + halfLen] - curImag * imag[i + k + halfLen]
                    val tImag = curReal * imag[i + k + halfLen] + curImag * real[i + k + halfLen]

                    real[i + k + halfLen] = real[i + k] - tReal
                    imag[i + k + halfLen] = imag[i + k] - tImag
                    real[i + k] = real[i + k] + tReal
                    imag[i + k] = imag[i + k] + tImag

                    val newCurReal = curReal * wReal - curImag * wImag
                    curImag = curReal * wImag + curImag * wReal
                    curReal = newCurReal
                }
                i += len
            }
            len *= 2
        }
    }

    private fun hzToMel(hz: Double): Double = 2595.0 * log10(1.0 + hz / 700.0)
    private fun melToHz(mel: Double): Double = 700.0 * (10.0.pow(mel / 2595.0) - 1.0)

    private fun buildMelFilterbank(): Array<FloatArray> {
        val lowMel = hzToMel(0.0)
        val highMel = hzToMel(sampleRate / 2.0)
        val melPoints = DoubleArray(numMelFilters + 2) { i ->
            lowMel + i * (highMel - lowMel) / (numMelFilters + 1)
        }
        val hzPoints = DoubleArray(melPoints.size) { melToHz(melPoints[it]) }
        val binPoints = IntArray(hzPoints.size) { i ->
            floor((nfft + 1) * hzPoints[i] / sampleRate).toInt()
        }

        val spectrumSize = nfft / 2 + 1
        return Array(numMelFilters) { m ->
            val filter = FloatArray(spectrumSize)
            val left = binPoints[m]
            val center = binPoints[m + 1]
            val right = binPoints[m + 2]

            for (k in left until center) {
                if (center > left) {
                    filter[k.coerceIn(0, spectrumSize - 1)] =
                        (k - left).toFloat() / (center - left)
                }
            }
            for (k in center until right) {
                if (right > center) {
                    filter[k.coerceIn(0, spectrumSize - 1)] =
                        (right - k).toFloat() / (right - center)
                }
            }
            filter
        }
    }

    private fun applyMelFilterbank(powerSpectrum: FloatArray): FloatArray {
        return FloatArray(numMelFilters) { m ->
            var sum = 0f
            for (k in powerSpectrum.indices) {
                sum += powerSpectrum[k] * melFilterbank[m][k]
            }
            sum
        }
    }

    private fun buildDctMatrix(): Array<FloatArray> {
        val normFactor = sqrt(2.0 / numMelFilters).toFloat()
        return Array(numMfcc) { i ->
            FloatArray(numMelFilters) { j ->
                normFactor * cos(PI * i * (2.0 * j + 1) / (2.0 * numMelFilters)).toFloat()
            }
        }
    }

    private fun applyDct(logMelEnergies: FloatArray): FloatArray {
        return FloatArray(numMfcc) { i ->
            var sum = 0f
            for (j in logMelEnergies.indices) {
                sum += dctMatrix[i][j] * logMelEnergies[j]
            }
            sum
        }
    }

    private fun padOrTruncate(
        mfcc: Array<FloatArray>,
        targetFrames: Int,
        targetCoeffs: Int
    ): Array<FloatArray> {
        return Array(targetFrames) { i ->
            if (i < mfcc.size) {
                FloatArray(targetCoeffs) { j ->
                    if (j < mfcc[i].size) mfcc[i][j] else 0f
                }
            } else {
                FloatArray(targetCoeffs)
            }
        }
    }
}
