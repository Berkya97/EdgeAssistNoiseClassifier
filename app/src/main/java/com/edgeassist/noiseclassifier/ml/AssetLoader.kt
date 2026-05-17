package com.edgeassist.noiseclassifier.ml

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName
import java.io.BufferedReader
import java.io.InputStreamReader

private const val TAG = "NoiseClassifier"

/**
 * norm_params.json format: {"x_min": float, "x_max": float}
 */
data class NormParams(
    @SerializedName("x_min") val xMin: Float,
    @SerializedName("x_max") val xMax: Float
) {
    fun normalize(value: Float): Float {
        return if (xMax - xMin > 1e-6f) {
            (value - xMin) / (xMax - xMin)
        } else {
            value
        }
    }
}

/**
 * Assets'ten JSON ve metin dosyalarını yükler.
 */
object AssetLoader {

    fun loadNormParams(context: Context, path: String = "norm_params.json"): NormParams? {
        return try {
            context.assets.open(path).use { input ->
                val reader = BufferedReader(InputStreamReader(input))
                val json = reader.readText()
                Gson().fromJson(json, NormParams::class.java)
            }.also { Log.d(TAG, "Loaded norm_params: x_min=${it?.xMin}, x_max=${it?.xMax}") }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load norm_params.json: ${e.message}")
            null
        }
    }

    fun loadLabels(context: Context, path: String = "labels.txt"): List<String> {
        return try {
            context.assets.open(path).use { input ->
                val reader = BufferedReader(InputStreamReader(input))
                reader.readLines().filter { it.isNotBlank() }
            }.also { Log.d(TAG, "Loaded ${it.size} labels from labels.txt") }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to load labels.txt: ${e.message}")
            emptyList()
        }
    }

    fun assetExists(context: Context, path: String): Boolean {
        return try {
            context.assets.open(path).close()
            true
        } catch (_: Exception) {
            false
        }
    }
}
