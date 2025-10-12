// =======================================================
//                   К О Н Ф И Г У Р А Ц И Я
// =======================================================
object Config {
    // Окно
    const val WIDTH_PX = 1200
    const val HEIGHT_PX = 800

    // Физика
    const val G = 80.0
    const val DT = 0.015
    const val SOFTENING = 1.0
    val SOFT2 = SOFTENING * SOFTENING
    var theta: Double = 0.35  // Barnes–Hut θ

    // Начальные количества/массы
    const val BODIES_COUNT = 2000
    const val CENTRAL_MASS = 50_000.0
    const val MIN_R = 24.0

    // Для «кеплера» — фиксируем общую массу диска, чтобы поведение не зависело от N
    const val TOTAL_SATELLITE_MASS = 1000.0

    // Для самогравитационного диска
    const val SELF_DISK_TOTAL_MASS = 6000.0
    const val SELF_DISK_CENTRAL_MASS = 600.0
    const val SELF_DISK_SPIRAL_EPS = 0.08
    const val SELF_DISK_Q_COLD = 0.08

    enum class Scene { KEPLER_DISK, SELF_GRAV_DISK }
    var scene: Scene = Scene.KEPLER_DISK
}