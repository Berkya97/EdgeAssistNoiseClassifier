package com.edgeassist.noiseclassifier

import android.Manifest
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.ui.platform.LocalContext
import androidx.core.content.FileProvider
import com.edgeassist.noiseclassifier.metrics.MetricsLogger
import com.edgeassist.noiseclassifier.metrics.SessionLogger
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Cloud
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.MicOff
import androidx.compose.material.icons.filled.PhoneAndroid
import androidx.compose.material.icons.filled.SyncAlt
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.edgeassist.noiseclassifier.data.model.ClassificationMode
import com.edgeassist.noiseclassifier.data.model.DebugFrame
import com.edgeassist.noiseclassifier.data.model.InferenceResult
import com.edgeassist.noiseclassifier.ui.MainViewModel
import com.edgeassist.noiseclassifier.ui.theme.EdgeAssistTheme
import com.edgeassist.noiseclassifier.ui.theme.GreenAccent
import com.edgeassist.noiseclassifier.ui.theme.RedAccent

class MainActivity : ComponentActivity() {

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { _ -> }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        setContent {
            EdgeAssistTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen()
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(viewModel: MainViewModel = viewModel()) {
    val isRecording by viewModel.isRecording.collectAsState()
    val selectedMode by viewModel.selectedMode.collectAsState()
    val latestResult by viewModel.latestResult.collectAsState()
    val isProcessing by viewModel.isProcessing.collectAsState()
    val inferenceCount by viewModel.inferenceCount.collectAsState()
    val logFileName by viewModel.logFileName.collectAsState()
    val lastSessionSummary by viewModel.lastSessionSummary.collectAsState()
    val context = LocalContext.current

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        "Edge-Assist Noise Classifier",
                        style = MaterialTheme.typography.titleLarge
                    )
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp)
                .verticalScroll(rememberScrollState()),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Spacer(modifier = Modifier.height(20.dp))

            ModeSelector(
                selectedMode = selectedMode,
                onModeSelected = { viewModel.setMode(it) },
                enabled = !isRecording
            )

            Spacer(modifier = Modifier.height(28.dp))

            RecordButton(
                isRecording = isRecording,
                onClick = { viewModel.toggleRecording() }
            )

            if (isRecording) {
                Spacer(modifier = Modifier.height(8.dp))
                RecordingIndicator()
                Text(
                    "Inferences: $inferenceCount",
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            if (isProcessing) {
                Spacer(modifier = Modifier.height(8.dp))
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 32.dp)
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            latestResult?.let { result ->
                ResultCard(result = result)
                result.debugFrame?.let { debug ->
                    Spacer(modifier = Modifier.height(8.dp))
                    DebugTuningLine(debug = debug)
                    Spacer(modifier = Modifier.height(8.dp))
                    DebugPanel(debug = debug)
                }
            } ?: PlaceholderCard()

            Spacer(modifier = Modifier.height(16.dp))

            lastSessionSummary?.let { summary ->
                if (!isRecording) {
                    SessionSummaryCard(summary = summary)
                    Spacer(modifier = Modifier.height(16.dp))
                }
            }

            LogExportSection(
                logFileName = logFileName,
                onExport = { shareCurrentCsv(context) }
            )

            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

@Composable
fun LogExportSection(logFileName: String, onExport: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(
            "Log: ${logFileName.ifEmpty { "(yok)" }}",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center
        )
        Spacer(modifier = Modifier.height(8.dp))
        Button(
            onClick = onExport,
            colors = ButtonDefaults.buttonColors(
                containerColor = MaterialTheme.colorScheme.secondaryContainer,
                contentColor = MaterialTheme.colorScheme.onSecondaryContainer
            )
        ) {
            Text("CSV Dışa Aktar")
        }
    }
}

private fun shareCurrentCsv(context: Context) {
    val file = MetricsLogger.getCurrentLogFile()
    if (file == null || !file.exists()) {
        Toast.makeText(context, "Log dosyası bulunamadı", Toast.LENGTH_SHORT).show()
        return
    }
    try {
        MetricsLogger.flush()
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file
        )
        val intent = Intent(Intent.ACTION_SEND).apply {
            type = "text/csv"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, file.name)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(intent, "CSV Paylaş"))
    } catch (e: Exception) {
        Toast.makeText(context, "Paylaşım başarısız: ${e.message}", Toast.LENGTH_LONG).show()
    }
}

@Composable
fun ModeSelector(
    selectedMode: ClassificationMode,
    onModeSelected: (ClassificationMode) -> Unit,
    enabled: Boolean
) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Text(
            "Inference Mode",
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Spacer(modifier = Modifier.height(8.dp))
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            ModeChip(
                mode = ClassificationMode.EDGE,
                icon = Icons.Filled.PhoneAndroid,
                isSelected = selectedMode == ClassificationMode.EDGE,
                onClick = { if (enabled) onModeSelected(ClassificationMode.EDGE) },
                modifier = Modifier.weight(1f)
            )
            ModeChip(
                mode = ClassificationMode.CLOUD,
                icon = Icons.Filled.Cloud,
                isSelected = selectedMode == ClassificationMode.CLOUD,
                onClick = { if (enabled) onModeSelected(ClassificationMode.CLOUD) },
                modifier = Modifier.weight(1f)
            )
            ModeChip(
                mode = ClassificationMode.ADAPTIVE,
                icon = Icons.Filled.SyncAlt,
                isSelected = selectedMode == ClassificationMode.ADAPTIVE,
                onClick = { if (enabled) onModeSelected(ClassificationMode.ADAPTIVE) },
                modifier = Modifier.weight(1f)
            )
        }
    }
}

@Composable
fun ModeChip(
    mode: ClassificationMode,
    icon: ImageVector,
    isSelected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val bgColor by animateColorAsState(
        targetValue = if (isSelected)
            MaterialTheme.colorScheme.primaryContainer
        else
            MaterialTheme.colorScheme.surfaceVariant,
        label = "modeChipBg"
    )
    val borderColor by animateColorAsState(
        targetValue = if (isSelected)
            MaterialTheme.colorScheme.primary
        else
            Color.Transparent,
        label = "modeChipBorder"
    )

    Surface(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .border(2.dp, borderColor, RoundedCornerShape(12.dp))
            .clickable(onClick = onClick),
        color = bgColor,
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier.padding(vertical = 12.dp, horizontal = 8.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = icon,
                contentDescription = mode.displayName,
                tint = if (isSelected)
                    MaterialTheme.colorScheme.primary
                else
                    MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(24.dp)
            )
            Spacer(modifier = Modifier.height(4.dp))
            Text(
                mode.displayName,
                style = MaterialTheme.typography.labelMedium,
                color = if (isSelected)
                    MaterialTheme.colorScheme.primary
                else
                    MaterialTheme.colorScheme.onSurfaceVariant,
                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
            )
        }
    }
}

@Composable
fun RecordButton(isRecording: Boolean, onClick: () -> Unit) {
    val bgColor by animateColorAsState(
        targetValue = if (isRecording) RedAccent else GreenAccent,
        label = "recordBtnBg"
    )

    val pulseTransition = rememberInfiniteTransition(label = "pulse")
    val scale by pulseTransition.animateFloat(
        initialValue = 1f,
        targetValue = if (isRecording) 1.08f else 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(600),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulseScale"
    )

    Box(
        modifier = Modifier
            .size(96.dp)
            .scale(scale)
            .clip(CircleShape)
            .background(bgColor)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center
    ) {
        Icon(
            imageVector = if (isRecording) Icons.Filled.MicOff else Icons.Filled.Mic,
            contentDescription = if (isRecording) "Stop" else "Start",
            tint = Color.White,
            modifier = Modifier.size(40.dp)
        )
    }
    Spacer(modifier = Modifier.height(8.dp))
    Text(
        text = if (isRecording) "Tap to Stop" else "Tap to Record",
        style = MaterialTheme.typography.bodyMedium,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}

@Composable
fun RecordingIndicator() {
    val transition = rememberInfiniteTransition(label = "recording")
    val alpha by transition.animateFloat(
        initialValue = 0.3f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(500),
            repeatMode = RepeatMode.Reverse
        ),
        label = "recordingAlpha"
    )

    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier.padding(vertical = 4.dp)
    ) {
        Box(
            modifier = Modifier
                .size(10.dp)
                .clip(CircleShape)
                .background(RedAccent.copy(alpha = alpha))
        )
        Spacer(modifier = Modifier.width(6.dp))
        Text(
            "Recording...",
            style = MaterialTheme.typography.labelMedium,
            color = RedAccent
        )
    }
}

@Composable
fun ResultCard(result: InferenceResult) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant
        )
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(
                "Classification Result",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(12.dp))

            Text(
                result.label.replaceFirstChar { it.uppercase() }.replace('_', ' '),
                style = MaterialTheme.typography.headlineLarge,
                color = MaterialTheme.colorScheme.primary
            )
            Spacer(modifier = Modifier.height(8.dp))

            Text(
                "Confidence",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(4.dp))
            LinearProgressIndicator(
                progress = result.confidence,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(8.dp)
                    .clip(RoundedCornerShape(4.dp)),
                color = MaterialTheme.colorScheme.primary,
                trackColor = MaterialTheme.colorScheme.surfaceVariant,
            )
            Text(
                "${(result.confidence * 100).toInt()}%",
                style = MaterialTheme.typography.labelMedium,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.End,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            Spacer(modifier = Modifier.height(16.dp))

            MetricRow("Mode Used", result.modeUsed.displayName)
            MetricRow("Total Latency", "${result.totalLatencyMs} ms")
            MetricRow("Inference Time", "${result.inferenceTimeMs} ms")
            if (result.bytesSent > 0) {
                MetricRow("Bytes Sent", "${result.bytesSent}")
            }
            if (result.batteryDelta != 0f) {
                MetricRow("Battery Delta", "%.2f%%".format(result.batteryDelta))
            }
        }
    }
}

@Composable
fun MetricRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSurface,
            fontSize = 14.sp
        )
    }
}

@Composable
fun DebugTuningLine(debug: DebugFrame) {
    Text(
        "p1=%.2f thresh=%.2f H=%.2f m=%.2f %s".format(
            debug.top1Prob,
            debug.adaptiveThreshold,
            debug.entropy,
            debug.top1Top2Margin,
            debug.unknownGatingReason ?: "—"
        ),
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.onSurfaceVariant,
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 4.dp)
    )
}

@Composable
fun DebugPanel(debug: DebugFrame) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f)
        )
    ) {
        Column(modifier = Modifier.padding(14.dp)) {
            Text(
                "Debug",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(8.dp))
            MetricRow("RMS", "%.5f".format(debug.rms))
            MetricRow("Mean mel (dB)", "%.1f".format(debug.meanMelEnergy))
            MetricRow("Entropy", "%.3f".format(debug.entropy))
            MetricRow("Top1–Top2 margin", "%.3f".format(debug.top1Top2Margin))
            MetricRow("Raw", "${debug.rawLabel} (%.2f)".format(debug.rawConfidence))
            MetricRow("Smoothed", "${debug.smoothedLabel} (%.2f)".format(debug.smoothedConfidence))
            if (debug.stickinessActive) {
                Text(
                    "Stickiness active",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
            debug.unknownGatingReason?.let { reason ->
                Text(
                    "Unknown reason: $reason",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
            Text(
                "Rolling mean Top3: " + debug.rollingMeanTop3.joinToString { "${it.first}=%.2f".format(it.second) },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 4.dp)
            )
            Text(
                "Frame Top3: " + debug.top3Labels.joinToString { "${it.first}=%.2f".format(it.second) },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(top = 2.dp)
            )
        }
    }
}

@Composable
fun SessionSummaryCard(summary: SessionLogger.SessionSummary) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 3.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer
        )
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Text(
                "Oturum Özeti",
                style = MaterialTheme.typography.labelLarge,
                color = MaterialTheme.colorScheme.onSecondaryContainer
            )
            Spacer(modifier = Modifier.height(12.dp))

            val durationSec = summary.durationSec
            val silencePct = summary.silenceRatio * 100.0
            val bytesSentKb = summary.totalBytesSent / 1024.0
            val bytesRecvKb = summary.totalBytesReceived / 1024.0

            SummaryRow("Mod", summary.mode)
            SummaryRow("Süre", "%.1f s".format(durationSec))

            val batteryText = if (summary.startBatteryPercent >= 0 && summary.endBatteryPercent >= 0) {
                "%%%d → %%%d (delta: -%d%%)".format(
                    summary.startBatteryPercent,
                    summary.endBatteryPercent,
                    summary.batteryDelta
                )
            } else {
                "(okunamadı)"
            }
            SummaryRow("Batarya", batteryText)

            if (summary.batteryDropPerMinute > 0.0) {
                SummaryRow("Dakika başına düşüş", "~%.2f %%/dk".format(summary.batteryDropPerMinute))
            }

            SummaryRow(
                "Toplam inference",
                "${summary.inferenceCount} (sessizlik: ${summary.silenceCount}, %.0f%%)".format(silencePct)
            )
            SummaryRow("Ortalama latency", "%.1f ms".format(summary.avgTotalLatencyMs))

            if (summary.totalBytesSent > 0 || summary.totalBytesReceived > 0) {
                SummaryRow(
                    "Toplam veri",
                    "gönderilen %.1f KB / alınan %.1f KB".format(bytesSentKb, bytesRecvKb)
                )
            }

            // Tahmini enerji (yalnızca anlamlı olduğunda göster)
            if (summary.inferenceCount > 0 && summary.durationMs > 60_000L && summary.batteryDelta > 0) {
                val perInference = SessionLogger.estimateEnergyPerInference(summary)
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    "İnference başına ~%.4f %% batarya".format(perInference),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSecondaryContainer
                )
            }
        }
    }
}

@Composable
private fun SummaryRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 3.dp),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSecondaryContainer
        )
        Text(
            value,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.onSecondaryContainer,
            fontSize = 14.sp
        )
    }
}

@Composable
fun PlaceholderCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
        )
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(
                imageVector = Icons.Filled.Mic,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.4f),
                modifier = Modifier.size(48.dp)
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                "Start recording to classify urban sounds",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                textAlign = TextAlign.Center
            )
        }
    }
}
