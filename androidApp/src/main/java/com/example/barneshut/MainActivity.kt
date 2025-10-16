package com.example.barneshut

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.tooling.preview.Preview
import barneshut.BodySnapshot
import barneshut.BodyFactory
import barneshut.BarnesHutSimulation
import barneshut.SimulationConfig
import androidx.compose.material3.darkColorScheme
import androidx.compose.ui.geometry.Offset
import kotlinx.coroutines.isActive

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val app = application as BarnesHutApplication
        setContent {
            BarnesHutTheme {
                BarnesHutSimulationScreen(app.config)
            }
        }
    }
}

@Composable
private fun BarnesHutSimulationScreen(config: SimulationConfig) {
    val simulationState = rememberSimulation(config)
    val bodiesState = simulationState.bodiesState
    LaunchedEffect(Unit) {
        while (isActive) {
            withFrameNanos {
                simulationState.step()
            }
        }
    }
    SimulationCanvas(config, bodiesState.value)
}

private class SimulationHolder(
    private val config: SimulationConfig
) {
    private val simulation = BarnesHutSimulation(
        config,
        BodyFactory.makeGalaxyDisk(config, config.bodiesPerCreation)
    )
    val bodiesState: MutableState<List<BodySnapshot>> = mutableStateOf(simulation.snapshot())

    fun step() {
        simulation.step()
        bodiesState.value = simulation.snapshot()
    }
}

@Composable
private fun rememberSimulation(config: SimulationConfig): SimulationHolder {
    return remember(config) { SimulationHolder(config) }
}

@Composable
private fun SimulationCanvas(config: SimulationConfig, bodies: List<BodySnapshot>) {
    Canvas(modifier = Modifier.fillMaxSize().background(Color.Black)) {
        val scaleX = if (config.widthPx > 0) size.width / config.widthPx.toFloat() else 1f
        val scaleY = if (config.heightPx > 0) size.height / config.heightPx.toFloat() else 1f
        for (body in bodies) {
            val x = (body.x * scaleX).toFloat()
            val y = (body.y * scaleY).toFloat()
            drawCircle(color = Color.White, radius = 2.0f, center = Offset(x, y))
        }
    }
}

@Composable
fun BarnesHutTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(),
        content = content
    )
}

@Preview
@Composable
private fun PreviewSimulation() {
    BarnesHutTheme {
        SimulationCanvas(SimulationConfig(), emptyList())
    }
}
