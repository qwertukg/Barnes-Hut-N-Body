import kotlin.math.*
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
        rng: Random = Random(3),
        vx: Double = 0.0,
        vy: Double = 0.0,
        x: Double = Config.WIDTH_PX * 0.5,
        y: Double = Config.HEIGHT_PX * 0.5,
        r: Double = min(Config.WIDTH_PX, Config.HEIGHT_PX) * 0.38
    ): MutableList<Body> {
        val cx = x
        val cy = y
        val rMax = r
        val sats = (nTotal - 1).coerceAtLeast(0)
        val bodies = ArrayList<Body>(sats + 1)

        bodies += Body(cx, cy, vx, vy, Config.CENTRAL_MASS)

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
            b.vx += vx
            b.vy += vy
        }
        return bodies
    }
}