// =======================================================
//                   К О Н Ф И Г У Р А Ц И Я
// =======================================================
object Config {
    // Окно
    var WIDTH_PX = 2400
    var HEIGHT_PX = 800

    // Физика
    const val COLLIDE_MAX_M = 10.0
    const val COLLIDE_MAX_V = 0.0
    const val COLLIDE_K = 0.1

    const val G = 80.0
    const val DT = 0.015
    const val SOFTENING = 1.0
    val SOFT2 = SOFTENING * SOFTENING
    var theta: Double = 0.40  // Barnes–Hut θ

    // Начальные количества/массы
    var BODIES_COUNT = 1000
    const val CENTRAL_MASS = 50_000.0
    const val MIN_R = 24.0

    // Для «кеплера» — фиксируем общую массу диска, чтобы поведение не зависело от N
    const val TOTAL_SATELLITE_MASS = 1000.0

    enum class Scene { KEPLER_DISK, KEPLER_DISK_COLLIDER }
    var scene: Scene = Scene.KEPLER_DISK
}