package com.edgeassist.noiseclassifier.ml

import android.util.Log
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.sin

/**
 * Hafif Mel Spectrogram çıkarıcı - sadece Kotlin FloatArray kullanır, native kütüphane yok.
 *
 * Librosa ile matematiksel uyum:
 * - Periodic Hann window (scipy get_window('hann', N, fftbins=True))
 * - Power spectrum: |FFT|² (librosa convention, n_fft bölmesi yok)
 * - Slaney mel skalası (htk=False) + Slaney filterbank norm
 * - power_to_db: 10*log10(power/ref), ref=max, amin=1e-10, top_db=80
 *
 * Eğitim ön işlemesi ile uyumlu:
 * - 16kHz mono, tam 1.0s (pad/trim)
 * - PCM 16-bit -> Float [-1, 1] (librosa.load ile aynı)
 * - n_mels=64, n_fft=1024, hop_length=512, center=False
 *
 * 1 saniye (16000 sample) için deterministik çıktı: (N_MELS, N_TIME_FRAMES) = (64, 30)
 */
class MelSpectrogramExtractor(
    private val nMels: Int = DEFAULT_N_MELS,
    private val nFft: Int = DEFAULT_N_FFT,
    private val hopLength: Int = DEFAULT_HOP_LENGTH,
    private val sampleRate: Int = DEFAULT_SAMPLE_RATE,
    private val targetSamples: Int = sampleRate * 1,
    private val enableDebugLogs: Boolean = false
) {

    companion object {
        private const val TAG = "NoiseClassifier"
        const val DEFAULT_N_MELS = 64
        const val DEFAULT_N_FFT = 1024
        const val DEFAULT_HOP_LENGTH = 512
        const val DEFAULT_SAMPLE_RATE = 16000

        /** PCM 16-bit -> Float [-1, 1] (librosa WAV yüklemesi ile uyumlu) */
        private const val PCM_SCALE = 1f / 32768f

        /** 1 saniye 16kHz için sabit zaman frame sayısı (center=False) */
        val N_TIME_FRAMES_1S: Int = 1 + (16000 - DEFAULT_N_FFT) / DEFAULT_HOP_LENGTH  // = 30
    }

    private val spectrumSize = nFft / 2 + 1
    /** Periodic Hann window — librosa/scipy get_window('hann', N, fftbins=True) ile uyumlu */
    private val hannWindow = FloatArray(nFft) { i ->
        (0.5 * (1.0 - cos(2.0 * PI * i / nFft))).toFloat()
    }

    private val melFilterbank: Array<FloatArray> by lazy { buildMelFilterbank() }

    /**
     * ShortArray (PCM 16-bit) alır, Mel Spectrogram döner.
     * Shape: (nMels, nTimeFrames) = (64, 30) - 1 saniye için deterministik.
     */
    fun extract(audioSamples: ShortArray): Array<FloatArray> {
        val startTime = System.nanoTime()

        val signal = padOrTrimToTarget(audioSamples)
        val frames = frameSignal(signal)

        val powerSpectra = Array(frames.size) { i ->
            computePowerSpectrum(applyWindow(frames[i]))
        }

        val melEnergies = Array(powerSpectra.size) { i ->
            applyMelFilterbank(powerSpectra[i])
        }

        val melDbTimeFirst = powerToDb(melEnergies) // shape: (nTimeFrames, nMels)

        // Transpose: (nTimeFrames, nMels) → (nMels, nTimeFrames) — librosa / train.py convention
        val nTime = melDbTimeFirst.size
        val melDb = Array(nMels) { m ->
            FloatArray(nTime) { t -> melDbTimeFirst[t][m] }
        }

        if (enableDebugLogs) {
            var melMin = Float.MAX_VALUE
            var melMax = Float.MIN_VALUE
            for (i in melDb.indices) for (j in melDb[i].indices) {
                melMin = minOf(melMin, melDb[i][j])
                melMax = maxOf(melMax, melDb[i][j])
            }
            Log.d(TAG, "Mel spectrogram before norm: min=%.2f max=%.2f dB".format(melMin, melMax))
        }

        val elapsedMs = (System.nanoTime() - startTime) / 1_000_000.0
        Log.d(TAG, "Mel extraction: ${audioSamples.size} samples -> [${melDb.size} x ${melDb[0].size}] in %.2f ms".format(elapsedMs))

        return melDb
    }

    private fun padOrTrimToTarget(samples: ShortArray): FloatArray {
        val out = FloatArray(targetSamples)
        when {
            samples.size >= targetSamples -> {
                val start = (samples.size - targetSamples) / 2
                for (i in 0 until targetSamples) {
                    out[i] = samples[start + i] * PCM_SCALE
                }
            }
            else -> {
                for (i in samples.indices) out[i] = samples[i] * PCM_SCALE
                for (i in samples.size until targetSamples) out[i] = 0f
            }
        }
        if (enableDebugLogs) {
            val min = out.minOrNull() ?: 0f
            val max = out.maxOrNull() ?: 0f
            Log.d(TAG, "PCM raw min=%.4f max=%.4f (scaled to [-1,1])".format(min, max))
        }
        return out
    }

    private fun frameSignal(signal: FloatArray): Array<FloatArray> {
        val nFrames = 1 + (signal.size - nFft) / hopLength
        return Array(nFrames) { i ->
            val start = i * hopLength
            FloatArray(nFft) { j ->
                if (start + j < signal.size) signal[start + j] else 0f
            }
        }
    }

    private fun applyWindow(frame: FloatArray): FloatArray {
        return FloatArray(nFft) { i -> frame[i] * hannWindow[i] }
    }

    private fun computePowerSpectrum(windowedFrame: FloatArray): FloatArray {
        val real = FloatArray(nFft) { if (it < windowedFrame.size) windowedFrame[it] else 0f }
        val imag = FloatArray(nFft)

        fft(real, imag)

        // librosa: |FFT|² (n_fft bölmesi yok)
        return FloatArray(spectrumSize) { i ->
            real[i] * real[i] + imag[i] * imag[i]
        }
    }

    private fun fft(real: FloatArray, imag: FloatArray) {
        val n = real.size
        if (n <= 1) return

        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                var t = real[i]; real[i] = real[j]; real[j] = t
                t = imag[i]; imag[i] = imag[j]; imag[j] = t
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
                var curReal = 1f
                var curImag = 0f

                for (k in 0 until halfLen) {
                    val tReal = curReal * real[i + k + halfLen] - curImag * imag[i + k + halfLen]
                    val tImag = curReal * imag[i + k + halfLen] + curImag * real[i + k + halfLen]

                    real[i + k + halfLen] = real[i + k] - tReal
                    imag[i + k + halfLen] = imag[i + k] - tImag
                    real[i + k] += tReal
                    imag[i + k] += tImag

                    val newCurReal = curReal * wReal - curImag * wImag
                    curImag = curReal * wImag + curImag * wReal
                    curReal = newCurReal
                }
                i += len
            }
            len *= 2
        }
    }

    /**
     * Slaney mel skalası — librosa default (htk=False).
     * <1000 Hz: lineer, >=1000 Hz: logaritmik.
     */
    private fun hzToMel(hz: Double): Double {
        val fSp = 200.0 / 3.0
        val minLogHz = 1000.0
        val minLogMel = minLogHz / fSp
        val logStep = kotlin.math.ln(6.4) / 27.0
        return if (hz < minLogHz) hz / fSp else minLogMel + kotlin.math.ln(hz / minLogHz) / logStep
    }

    private fun melToHz(mel: Double): Double {
        val fSp = 200.0 / 3.0
        val minLogHz = 1000.0
        val minLogMel = minLogHz / fSp
        val logStep = kotlin.math.ln(6.4) / 27.0
        return if (mel < minLogMel) mel * fSp else minLogHz * kotlin.math.exp(logStep * (mel - minLogMel))
    }

    /**
     * librosa.filters.mel(norm='slaney') ile birebir uyumlu mel filterbank.
     *
     * Librosa akışı:
     *   1) mel_f = mel_frequencies(n_mels+2, fmin, fmax)   — Hz cinsinden
     *   2) fftfreqs = fft_frequencies(sr, n_fft)            — Hz cinsinden
     *   3) ramps[i,k] = mel_f[i] - fftfreqs[k]
     *   4) lower = -ramps[i] / fdiff[i],  upper = ramps[i+2] / fdiff[i+1]
     *   5) weights[i] = max(0, min(lower, upper))
     *   6) Slaney norm: weights[i] *= 2 / (mel_f[i+2] - mel_f[i])
     */
    private fun buildMelFilterbank(): Array<FloatArray> {
        val fftFreqs = DoubleArray(spectrumSize) { k ->
            k.toDouble() * sampleRate / nFft
        }

        val lowMel = hzToMel(0.0)
        val highMel = hzToMel(sampleRate / 2.0)
        val melF = DoubleArray(nMels + 2) { i ->
            melToHz(lowMel + i * (highMel - lowMel) / (nMels + 1))
        }

        val fdiff = DoubleArray(nMels + 1) { i -> melF[i + 1] - melF[i] }

        val ramps = Array(nMels + 2) { i ->
            DoubleArray(spectrumSize) { k -> melF[i] - fftFreqs[k] }
        }

        return Array(nMels) { m ->
            val filter = FloatArray(spectrumSize) { k ->
                val lower = -ramps[m][k] / fdiff[m]
                val upper = ramps[m + 2][k] / fdiff[m + 1]
                maxOf(0.0, minOf(lower, upper)).toFloat()
            }
            // Slaney norm: 2 / (mel_f[m+2] - mel_f[m])
            val bandwidthHz = melF[m + 2] - melF[m]
            if (bandwidthHz > 1e-10) {
                val enorm = (2.0 / bandwidthHz).toFloat()
                for (k in filter.indices) filter[k] *= enorm
            }
            filter
        }
    }

    private fun applyMelFilterbank(powerSpectrum: FloatArray): FloatArray {
        return FloatArray(nMels) { m ->
            var sum = 0f
            for (k in powerSpectrum.indices) {
                sum += powerSpectrum[k] * melFilterbank[m][k]
            }
            sum
        }
    }

    /**
     * Power spectrogram -> dB (log scale).
     * librosa.power_to_db(S, ref=np.max, amin=1e-10, top_db=80.0) ile uyumlu.
     */
    private fun powerToDb(melEnergies: Array<FloatArray>): Array<FloatArray> {
        val amin = 1e-10
        var globalMax = amin
        for (i in melEnergies.indices) {
            for (j in melEnergies[i].indices) {
                globalMax = max(globalMax, melEnergies[i][j].toDouble())
            }
        }
        val refValue = max(amin, globalMax)

        val result = Array(melEnergies.size) { i ->
            FloatArray(melEnergies[i].size) { j ->
                val p = melEnergies[i][j].toDouble().coerceAtLeast(amin)
                (10.0 * log10(p / refValue)).toFloat()
            }
        }

        // top_db=80 kırpma (librosa default)
        val topDb = 80f
        var dbMax = -Float.MAX_VALUE
        for (row in result) for (v in row) if (v > dbMax) dbMax = v
        val threshold = dbMax - topDb
        for (i in result.indices) for (j in result[i].indices) {
            if (result[i][j] < threshold) result[i][j] = threshold
        }

        return result
    }
}
