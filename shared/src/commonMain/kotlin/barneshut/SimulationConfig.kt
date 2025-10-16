package barneshut

/**
 * Параметры симуляции Barnes–Hut, вынесенные в data-класс для мультиплатформенности.
 * Значения подобраны в духе канона DAML: холодный диск с мягким сглаживанием.
 */
data class SimulationConfig(
    val widthPx: Double = 2400.0,
    val heightPx: Double = 800.0,
    val g: Double = 80.0,
    val dt: Double = 0.005,
    val softening: Double = 1.0,
    val theta: Double = 0.30,
    val creationRadius: Double = 100.0,
    val bodiesPerCreation: Int = 5_000,
    val centralMass: Double = 50_000.0,
    val minRadius: Double = 8.0,
    val totalSatelliteMass: Double = 5_000.0
) {
    val softeningSquared: Double = softening * softening
}
