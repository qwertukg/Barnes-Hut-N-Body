import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

// =======================================================
//                   С О З Д А Н И Е   Т Е Л
//             (сцены/раскладки и скорости)
// =======================================================
object BodyFactory {

    /** Кеплеровский диск: центр + спутники с v_circ по M_enclosed(r). */
    fun makeKeplerDisk(
        nTotal: Int,
        clockwise: Boolean = true,
        radialJitter: Double = 0.03,
        speedJitter: Double = 0.01,
        rng: Random = Random(3)
    ): MutableList<Body> {
        val cx = Config.WIDTH_PX * 0.5
        val cy = Config.HEIGHT_PX * 0.5
        val rMax = min(Config.WIDTH_PX, Config.HEIGHT_PX) * 0.38
        val sats = (nTotal - 1).coerceAtLeast(0)

        val bodies = ArrayList<Body>(sats + 1)
        bodies += Body(cx, cy, 0.0, 0.0, Config.CENTRAL_MASS)

        val mSat = if (sats > 0) Config.TOTAL_SATELLITE_MASS / sats else 0.0

        repeat(sats) {
            val u = rng.nextDouble()
            val rr = sqrt(u * (rMax * rMax - Config.MIN_R * Config.MIN_R) + Config.MIN_R * Config.MIN_R)
            val rJ = rr * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * radialJitter)
            val ang = rng.nextDouble() * 2.0 * Math.PI
            val x = cx + rJ * cos(ang)
            val y = cy + rJ * sin(ang)
            bodies += Body(x, y, 0.0, 0.0, mSat)
        }

        data class RIdx(val i: Int, val r: Double)
        val sorted = bodies.mapIndexed { i, b -> RIdx(i, hypot(b.x - cx, b.y - cy)) }.sortedBy { it.r }
        var acc = 0.0
        val Menc = DoubleArray(bodies.size)
        for (ri in sorted) { acc += bodies[ri.i].m; Menc[ri.i] = acc }

        for (i in 1 until bodies.size) {
            val b = bodies[i]
            val dx = b.x - cx; val dy = b.y - cy
            val r = max(1e-6, hypot(dx, dy))
            val vCirc = sqrt(Config.G * Menc[i] / r)
            val v = vCirc * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * speedJitter)
            val (tx, ty) = if (clockwise) (dy / r) to (-dx / r) else (-dy / r) to (dx / r)
            b.vx = tx * v; b.vy = ty * v
        }
        return bodies
    }

    /** Самогравитирующийся диск с семечком m=2 и небольшой дисперсией скоростей. */
    fun makeSelfGravDisk(
        nTotal: Int,
        rMin: Double = 20.0,
        clockwise: Boolean = true,
        spiralEps: Double = Config.SELF_DISK_SPIRAL_EPS,
        qCold: Double = Config.SELF_DISK_Q_COLD,
        rng: Random = Random(9)
    ): MutableList<Body> {
        val cx = Config.WIDTH_PX * 0.5
        val cy = Config.HEIGHT_PX * 0.5
        val rMax = min(Config.WIDTH_PX, Config.HEIGHT_PX) * 0.38
        val sats = (nTotal - 1).coerceAtLeast(0)
        val bodies = ArrayList<Body>(sats + 1)

        // Центр (небольшой, чтобы диск сам тянул)
        bodies += Body(cx, cy, 0.0, 0.0, Config.SELF_DISK_CENTRAL_MASS)

        val mSat = if (sats > 0) Config.SELF_DISK_TOTAL_MASS / sats else 0.0

        fun gauss(): Double {
            val u1 = rng.nextDouble().coerceIn(1e-12, 1.0)
            val u2 = rng.nextDouble()
            return sqrt(-2.0 * ln(u1)) * cos(2.0 * Math.PI * u2)
        }

        repeat(sats) {
            val u = rng.nextDouble()
            val baseR = sqrt(u * (rMax * rMax - rMin * rMin) + rMin * rMin)
            val phi = rng.nextDouble() * 2.0 * Math.PI
            val r = (baseR * (1.0 + spiralEps * cos(2.0 * phi))).coerceIn(rMin, rMax)
            val x = cx + r * cos(phi)
            val y = cy + r * sin(phi)
            bodies += Body(x, y, 0.0, 0.0, mSat)
        }

        data class RIdx(val i: Int, val r: Double)
        val sorted = bodies.mapIndexed { i, b -> RIdx(i, hypot(b.x - cx, b.y - cy)) }.sortedBy { it.r }
        var acc = 0.0
        val Menc = DoubleArray(bodies.size)
        for (ri in sorted) { acc += bodies[ri.i].m; Menc[ri.i] = acc }

        for (i in 1 until bodies.size) {
            val b = bodies[i]
            val dx = b.x - cx; val dy = b.y - cy
            val r = max(1e-6, hypot(dx, dy))
            val vCirc = sqrt(Config.G * Menc[i] / r)

            val rx = dx / r; val ry = dy / r
            val (tx, ty) = if (clockwise) ry to -rx else -ry to rx

            val sigmaR = qCold * vCirc
            val sigmaT = 0.5 * sigmaR
            val dvR = sigmaR * gauss()
            val dvT = sigmaT * gauss()

            b.vx = tx * (vCirc + dvT) + rx * dvR
            b.vy = ty * (vCirc + dvT) + ry * dvR
        }
        return bodies
    }
}