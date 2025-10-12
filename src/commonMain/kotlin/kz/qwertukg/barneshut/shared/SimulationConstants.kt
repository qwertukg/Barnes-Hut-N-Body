package kz.qwertukg.barneshut.shared

/**
 * Константы, определяющие физические параметры симуляции по канону DAML.
 */
object SimulationConstants {
    /** Масса центрального тела кеплеровского диска. */
    const val CENTRAL_MASS: Double = 50_000.0

    /** Минимальный радиус спутника от центра диска. */
    const val MIN_RADIUS: Double = 24.0

    /** Суммарная масса спутников в одном диске. */
    const val TOTAL_SATELLITE_MASS: Double = 1_000.0

    /** Минимальное значение параметра θ. */
    const val THETA_MIN: Double = 0.2

    /** Максимальное значение параметра θ. */
    const val THETA_MAX: Double = 1.6

    /** Минимальный радиус диска, создаваемого пользователем. */
    const val DISK_RADIUS_MIN: Double = 100.0

    /** Максимальный радиус диска, создаваемого пользователем. */
    const val DISK_RADIUS_MAX: Double = 500.0

    /** Минимальное количество тел в создаваемом диске. */
    const val DISK_BODIES_MIN: Int = 1_000

    /** Максимальное количество тел в создаваемом диске. */
    const val DISK_BODIES_MAX: Int = 10_000

    /** Минимальный шаг интегрирования по времени. */
    const val DT_MIN: Double = -0.015

    /** Максимальный шаг интегрирования по времени. */
    const val DT_MAX: Double = 0.015

    /** Минимальное значение гравитационной постоянной. */
    const val GRAVITY_MIN: Double = 0.0

    /** Максимальное значение гравитационной постоянной. */
    const val GRAVITY_MAX: Double = 100.0
}
