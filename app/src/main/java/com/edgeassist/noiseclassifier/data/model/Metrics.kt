package com.edgeassist.noiseclassifier.data.model

data class Metrics(
    val rttMs: Long = Long.MAX_VALUE,
    val batteryLevel: Int = 100,
    val cpuLoad: Float = 0f
)
