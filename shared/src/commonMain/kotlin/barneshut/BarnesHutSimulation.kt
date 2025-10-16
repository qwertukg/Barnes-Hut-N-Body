package barneshut

import kotlin.math.max

/**
 * Пошаговая симуляция Barnes–Hut, пригодная для Kotlin Multiplatform.
 * Алгоритм следует канону DAML: каждую итерацию строим квадродерево и считаем силы с учётом тэта-критерия.
 */
class BarnesHutSimulation(
    private val config: SimulationConfig,
    initialBodies: MutableList<Body>
) {
    private val bodies: MutableList<Body> = initialBodies
    private val acc = ForceAcc()

    fun bodies(): List<Body> = bodies

    fun step() {
        if (bodies.isEmpty()) return
        val h = max(config.widthPx, config.heightPx) * 0.5
        val root = BHTree(
            Quad(
                cx = config.widthPx * 0.5,
                cy = config.heightPx * 0.5,
                halfSize = max(h, 1.0)
            )
        )
        for (body in bodies) {
            root.insert(body)
        }
        root.computeMass()
        for (body in bodies) {
            acc.reset()
            root.accumulateForce(body, config, acc)
            val ax = acc.fx / body.m
            val ay = acc.fy / body.m
            body.vx += ax * config.dt
            body.vy += ay * config.dt
        }
        for (body in bodies) {
            body.x += body.vx * config.dt
            body.y += body.vy * config.dt
        }
        confineBodies()
    }

    private fun confineBodies() {
        val maxX = config.widthPx
        val maxY = config.heightPx
        for (body in bodies) {
            body.x = body.x.coerceIn(0.0, maxX)
            body.y = body.y.coerceIn(0.0, maxY)
        }
    }

    fun snapshot(): List<BodySnapshot> = bodies.map { BodySnapshot(it.x, it.y, it.vx, it.vy, it.m) }
}

/** Небольшой иммутабельный снимок тела для UI. */
data class BodySnapshot(
    val x: Double,
    val y: Double,
    val vx: Double,
    val vy: Double,
    val mass: Double
)
