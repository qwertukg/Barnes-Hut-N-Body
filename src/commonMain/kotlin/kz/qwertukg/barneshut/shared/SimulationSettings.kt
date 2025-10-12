package kz.qwertukg.barneshut.shared

import kotlinx.serialization.Serializable

/**
 * Текущие настраиваемые параметры симуляции Barnes–Hut.
 */
@Serializable
data class SimulationSettings(
    val widthPx: Int = 1920,
    val heightPx: Int = 1080,
    val gravity: Double = 80.0,
    val timeStep: Double = 0.015,
    val softening: Double = 1.0,
    val theta: Double = 0.40,
    val diskRadius: Double = 100.0,
    val bodiesPerDisk: Int = 2_000
) {
    /** Квадрат параметра сглаживания для ускоренного доступа. */
    val softeningSquared: Double get() = softening * softening
}
