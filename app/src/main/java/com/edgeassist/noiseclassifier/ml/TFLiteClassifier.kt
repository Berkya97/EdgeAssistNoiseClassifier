package com.edgeassist.noiseclassifier.ml

import android.content.Context
import android.util.Log
import com.edgeassist.noiseclassifier.data.model.ClassificationMode
import com.edgeassist.noiseclassifier.data.model.InferenceResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.random.Random

/**
 * TensorFlow Lite classifier for urban sound classification.
 *
 * Expected model input:  [1, 64, T, 1] float32 (mel spectrogram, channels-last)
 * Expected model output: [1, numClasses] float32 (softmax probabilities)
 *
 * Mel features: Array<FloatArray> shape (nMels, nTimeFrames) = (64, 30)
 * - Min-max normalizasyon: assets/norm_params.json (x_min, x_max)
 * - Etiketler: assets/labels.txt
 *
 * model.tflite, labels.txt veya norm_params.json yoksa dummy moda düşer.
 */
class TFLiteClassifier(
    private val context: Context,
    private val enableDebugLogs: Boolean = false
) {

    companion object {
        private const val TAG = "NoiseClassifier"
        private const val MODEL_FILE = "model.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val NORM_PARAMS_FILE = "norm_params.json"

        /** Dummy modda kullanılacak fallback etiketler (UrbanSound8K sınıfları) */
        private val DUMMY_LABELS = listOf(
            "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
            "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"
        )
    }

    private var interpreter: Interpreter? = null
    private var isModelLoaded = false
    private var labels: List<String> = emptyList()
    private var normParams: NormParams? = null

    /** Teşhis: aynı instance initialize vs classify'da mı kullanılıyor */
    private val instanceId = "TFLite@${Integer.toHexString(System.identityHashCode(this))}"

    fun initialize() {
        Log.w(TAG, "TFLiteClassifier.initialize() CALLED [id=$instanceId] — model load will follow")
        Log.d(TAG, "[$instanceId] initialize() start")
        labels = AssetLoader.loadLabels(context, LABELS_FILE)
        if (labels.isEmpty()) labels = DUMMY_LABELS
        Log.d(TAG, "[$instanceId] labels loaded: ${labels.size}")

        normParams = AssetLoader.loadNormParams(context, NORM_PARAMS_FILE)
        Log.d(TAG, "[$instanceId] normParams: ${if (normParams != null) "loaded (x_min=${normParams!!.xMin}, x_max=${normParams!!.xMax})" else "missing"}")

        // 1) Dosya adı + assets listesi (isim/boşluk/büyük-küçük hatası teşhisi) — TAG ile NoiseClassifier filtresinde görünsün
        val modelFileName = MODEL_FILE
        Log.d(TAG, "Model asset name used in code = '$modelFileName'")
        Log.d(TAG, "Assets list = ${context.assets.list("")?.toList()}")

        try {
            Log.d(TAG, "Loading TFLite model from assets: $modelFileName")
            val modelBuffer = try {
                loadModelMapped(context, modelFileName)
            } catch (e: Exception) {
                Log.w(TAG, "Mapped load failed, falling back to stream load", e)
                loadModelByteBuffer(context, modelFileName)
            }
            val options = Interpreter.Options().apply { setNumThreads(4) }
            interpreter = Interpreter(modelBuffer, options)
            isModelLoaded = true
            Log.i(TAG, "TFLite loaded OK. interpreter=$interpreter")
            val inputTensor = interpreter!!.getInputTensor(0)
            val outputTensor = interpreter!!.getOutputTensor(0)
            Log.d(TAG, "[$instanceId]   Input shape:  ${inputTensor.shape().contentToString()}")
            Log.d(TAG, "[$instanceId]   Output shape: ${outputTensor.shape().contentToString()}")
        } catch (e: Exception) {
            isModelLoaded = false
            interpreter = null
            Log.e(TAG, "TFLite load FAILED", e)
        }
        Log.d(TAG, "[$instanceId] initialize() end — init summary: isModelLoaded=$isModelLoaded, interpreter=${interpreter != null}, normParams=${normParams != null}")
    }

    /**
     * Mel spectrogram özellikleri ile sınıflandırma.
     * melFeatures: (nMels, nTimeFrames) = (64, 30)
     */
    fun classify(melFeatures: Array<FloatArray>): InferenceResult {
        val useDummy = !isModelLoaded || interpreter == null || normParams == null
        Log.d(TAG, "[$instanceId] classify() entry mode=${if (useDummy) "DUMMY" else "TFLITE"} (isModelLoaded=$isModelLoaded, interpreter=${interpreter != null}, normParams=${normParams != null})")
        if (useDummy) {
            Log.w(TAG, "[$instanceId] Using dummy classifier: isModelLoaded=$isModelLoaded, interpreter=${interpreter != null}, normParams=${normParams != null}")
            return classifyDummy()
        }
        return try {
            classifyWithModel(melFeatures)
        } catch (e: Exception) {
            Log.e(TAG, "[$instanceId] TFLite inference failed, falling back to dummy: ${e.message}", e)
            classifyDummy()
        }
    }

    private fun classifyWithModel(melFeatures: Array<FloatArray>): InferenceResult {
        Log.d(TAG, "[$instanceId] classifyWithModel() entry mode=TFLITE")
        val totalStart = System.nanoTime()
        val preprocessStart = System.nanoTime()

        val inputTensor = interpreter!!.getInputTensor(0)
        val inputShape = inputTensor.shape()
        val batchSize = inputShape[0]
        val height = inputShape[1]
        val width = inputShape[2]
        val channels = inputShape[3]

        Log.d(TAG, "Model input shape: $batchSize x $height x $width x $channels")

        // ----- 1) MEL (inference'e giden AYNI melFeatures) istatistikleri -----
        val nMels = melFeatures.size
        val nTime = if (melFeatures.isNotEmpty()) melFeatures[0].size else 0
        var melMin = Float.MAX_VALUE
        var melMax = Float.MIN_VALUE
        var melSum = 0.0
        var melCount = 0
        for (m in melFeatures.indices) for (t in melFeatures[m].indices) {
            val v = melFeatures[m][t]
            melMin = minOf(melMin, v)
            melMax = maxOf(melMax, v)
            melSum += v
            melCount++
        }
        val melMean = if (melCount > 0) melSum / melCount else 0f
        Log.d(TAG, "[MEL in] shape=${nMels}x${nTime} min=%.4f max=%.4f mean=%.4f".format(melMin, melMax, melMean))
        val melFirst10 = mutableListOf<Float>()
        for (m in 0 until minOf(nMels, 1)) for (t in 0 until minOf(10, nTime)) {
            melFirst10.add(melFeatures[m][t])
        }
        if (nMels > 1 && nTime < 10) for (m in 1 until nMels) for (t in 0 until nTime) {
            if (melFirst10.size >= 10) break
            melFirst10.add(melFeatures[m][t])
        }
        Log.d(TAG, "[MEL in] first 10 (m=0,t=0..9): ${melFirst10.take(10).joinToString { "%.4f".format(it) }}")
        Log.d(TAG, "[MEL in] (0,0)=%.4f (0,1)=%.4f (0,29)=%.4f (1,0)=%.4f".format(
            melFeatures.getOrNull(0)?.getOrNull(0) ?: 0f,
            melFeatures.getOrNull(0)?.getOrNull(1) ?: 0f,
            melFeatures.getOrNull(0)?.getOrNull(29) ?: 0f,
            melFeatures.getOrNull(1)?.getOrNull(0) ?: 0f
        ))

        val params = normParams!!
        val xMin = params.xMin
        val xMax = params.xMax
        val range = xMax - xMin
        Log.d(TAG, "norm_params: x_min=%.6f x_max=%.6f range=%.6f (train uses (x-x_min)/(x_max-x_min))".format(xMin, xMax, range))
        if (kotlin.math.abs(range) < 1e-5f) {
            Log.w(TAG, "norm_params range nearly zero — input will be constant or invalid!")
        }

        // ----- 2) Normalize et, önce FloatArray'te topla (buffer'dan önce log) -----
        val totalElements = batchSize * height * width * channels
        val normalizedFeatures = FloatArray(totalElements)
        var idx = 0
        for (m in 0 until height) {
            for (t in 0 until width) {
                val value = if (m < melFeatures.size && t < melFeatures[m].size) {
                    params.normalize(melFeatures[m][t])
                } else {
                    0f
                }
                normalizedFeatures[idx++] = value
            }
        }
        val normMin = normalizedFeatures.minOrNull() ?: 0f
        val normMax = normalizedFeatures.maxOrNull() ?: 0f
        val normMean = normalizedFeatures.average().toFloat()
        Log.d(TAG, "[normalizedFeatures] min=%.6f max=%.6f mean=%.6f".format(normMin, normMax, normMean))
        Log.d(TAG, "[normalizedFeatures] first 10: ${normalizedFeatures.take(10).joinToString { "%.4f".format(it) }}")
        Log.d(TAG, "[normalizedFeatures] last 10: ${normalizedFeatures.takeLast(10).joinToString { "%.4f".format(it) }}")
        Log.d(TAG, "[normalizedFeatures] (m,t)=(0,0),(0,1),(0,29),(1,0): %.4f %.4f %.4f %.4f".format(
            normalizedFeatures.getOrNull(0) ?: 0f,
            normalizedFeatures.getOrNull(1) ?: 0f,
            normalizedFeatures.getOrNull(29) ?: 0f,
            normalizedFeatures.getOrNull(30) ?: 0f
        ))

        val inputBuffer = ByteBuffer.allocateDirect(totalElements * 4)
            .order(ByteOrder.nativeOrder())
        for (v in normalizedFeatures) inputBuffer.putFloat(v)
        inputBuffer.rewind()

        // ----- 3) ByteBuffer'dan geri oku; first 10 ve last 10 -----
        val readFirst10 = FloatArray(10) { inputBuffer.float }
        inputBuffer.position((totalElements - 10) * 4)
        val readLast10 = FloatArray(10) { inputBuffer.float }
        inputBuffer.rewind()
        Log.d(TAG, "[ByteBuffer readback] first 10: ${readFirst10.joinToString { "%.4f".format(it) }}")
        Log.d(TAG, "[ByteBuffer readback] last 10: ${readLast10.joinToString { "%.4f".format(it) }}")

        val preprocessMs = (System.nanoTime() - preprocessStart) / 1_000_000
        Log.d(TAG, "Preprocessing (normalize + reshape): ${preprocessMs}ms")

        val outputTensor = interpreter!!.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val numClasses = outputShape[1]
        Log.d(TAG, "Model output shape: ${outputShape.contentToString()}")

        val outputBuffer = ByteBuffer.allocateDirect(numClasses * 4)
            .order(ByteOrder.nativeOrder())

        val inferenceStart = System.nanoTime()
        interpreter!!.run(inputBuffer, outputBuffer)
        val inferenceMs = (System.nanoTime() - inferenceStart) / 1_000_000
        outputBuffer.rewind()

        val probabilities = FloatArray(numClasses) { outputBuffer.float }

        var maxIdx = 0
        var maxProb = probabilities[0]
        for (i in 1 until numClasses) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIdx = i
            }
        }

        val label = if (maxIdx < labels.size) labels[maxIdx] else "unknown_$maxIdx"
        val totalMs = (System.nanoTime() - totalStart) / 1_000_000

        Log.d(TAG, "[$instanceId] mode=TFLITE — TFLite inference: label=$label, confidence=%.4f".format(maxProb))
        Log.d(TAG, "  Inference time: ${inferenceMs}ms, preprocessing: ${preprocessMs}ms, total: ${totalMs}ms")
        Log.d(TAG, "  Probabilities: ${probabilities.take(5).map { "%.4f".format(it) }}...")

        return InferenceResult(
            label = label,
            confidence = maxProb,
            inferenceTimeMs = inferenceMs,
            totalLatencyMs = totalMs,
            modeUsed = ClassificationMode.EDGE,
            probabilities = probabilities
        )
    }

    private fun classifyDummy(): InferenceResult {
        Log.d(TAG, "[$instanceId] classifyDummy() entry mode=DUMMY")
        val startTime = System.nanoTime()
        val n = labels.size.coerceAtLeast(10)
        val labelIndex = if (labels.isNotEmpty()) Random.nextInt(labels.size) else 0
        val label = if (labels.isNotEmpty()) labels[labelIndex] else "unknown"
        val confidence = Random.nextFloat() * 0.4f + 0.6f

        // One-hot-like probabilities so Temporal never sees null; confidence = probabilities[labelIndex]
        val probabilities = FloatArray(n) { i ->
            if (i == labelIndex) confidence else (1f - confidence) / (n - 1).coerceAtLeast(1)
        }

        val inferenceMs = (System.nanoTime() - startTime) / 1_000_000
        Log.d(TAG, "[$instanceId] mode=DUMMY — Dummy inference: label=$label, confidence=%.3f, time=${inferenceMs}ms, probabilities.size=$n".format(confidence))

        return InferenceResult(
            label = label,
            confidence = confidence,
            inferenceTimeMs = inferenceMs,
            totalLatencyMs = inferenceMs,
            modeUsed = ClassificationMode.EDGE,
            probabilities = probabilities
        )
    }

    /** openFd ile map — sıkıştırılmamış asset'te çalışır; compressed ise patlar. */
    private fun loadModelMapped(context: Context, fileName: String): MappedByteBuffer {
        context.assets.openFd(fileName).use { fd ->
            FileInputStream(fd.fileDescriptor).channel.use { ch ->
                return ch.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
    }

    /** Compressed asset fallback: stream oku, DirectByteBuffer'a yaz. */
    private fun loadModelByteBuffer(context: Context, fileName: String): ByteBuffer {
        context.assets.open(fileName).use { input ->
            val bytes = input.readBytes()
            return ByteBuffer.allocateDirect(bytes.size).apply {
                order(ByteOrder.nativeOrder())
                put(bytes)
                rewind()
            }
        }
    }

    fun getLabels(): List<String> = labels

    fun close() {
        interpreter?.close()
        interpreter = null
        isModelLoaded = false
        Log.d(TAG, "TFLiteClassifier closed")
    }
}
