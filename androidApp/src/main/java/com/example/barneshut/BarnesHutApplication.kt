package com.example.barneshut

import android.app.Application
import barneshut.SimulationConfig

/**
 * Application, удерживающее конфигурацию симуляции для Android.
 * Можно расширять, если потребуется управлять жизненным циклом.
 */
class BarnesHutApplication : Application() {
    val config: SimulationConfig by lazy { SimulationConfig(widthPx = 1920.0, heightPx = 1080.0) }
}
