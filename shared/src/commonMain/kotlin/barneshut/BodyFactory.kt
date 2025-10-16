package barneshut

import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.hypot
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

/** Генератор стартовых распределений масс по канону DAML. */
object BodyFactory {
    fun makeKeplerDisk(
        config: SimulationConfig,
        nTotal: Int,
        clockwise: Boolean = true,
        radialJitter: Double = 0.03,
        speedJitter: Double = 0.01,
        rng: Random = Random(3),
        vx: Double = 0.0,
        vy: Double = 0.0,
        x: Double = config.widthPx * 0.5,
        y: Double = config.heightPx * 0.5,
        r: Double = min(config.widthPx, config.heightPx) * 0.38
    ): MutableList<Body> {
        val cx = x
        val cy = y
        val rMax = r
        val sats = (nTotal - 1).coerceAtLeast(0)
        val bodies = ArrayList<Body>(sats + 1)
        bodies += Body(cx, cy, vx, vy, config.centralMass)
        val mSat = if (sats > 0) config.totalSatelliteMass / sats else 0.0
        repeat(sats) {
            val u = rng.nextDouble()
            val rr = sqrt(u * (rMax * rMax - config.minRadius * config.minRadius) + config.minRadius * config.minRadius)
            val rJ = rr * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * radialJitter)
            val ang = rng.nextDouble() * 2.0 * Math.PI
            val px = cx + rJ * cos(ang)
            val py = cy + rJ * sin(ang)
            bodies += Body(px, py, 0.0, 0.0, mSat)
        }
        data class RIdx(val index: Int, val radius: Double)
        val sorted = bodies.mapIndexed { i, b -> RIdx(i, hypot(b.x - cx, b.y - cy)) }.sortedBy { it.radius }
        var acc = 0.0
        val enclosed = DoubleArray(bodies.size)
        for (ri in sorted) {
            acc += bodies[ri.index].m
            enclosed[ri.index] = acc
        }
        for (i in 1 until bodies.size) {
            val body = bodies[i]
            val dx = body.x - cx
            val dy = body.y - cy
            val radius = max(1e-6, hypot(dx, dy))
            val vCirc = sqrt(config.g * enclosed[i] / radius)
            val v = vCirc * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * speedJitter)
            val tangent = if (clockwise) Pair(dy / radius, -dx / radius) else Pair(-dy / radius, dx / radius)
            body.vx = tangent.first * v + vx
            body.vy = tangent.second * v + vy
        }
        return bodies
    }

    fun makeGalaxyDisk(
        config: SimulationConfig,
        nTotal: Int,
        epsM2: Double = 0.03,
        phi0: Double = 0.0,
        barTaperR: Double? = null,
        radialScale: Double? = null,
        speedJitter: Double = 0.01,
        radialJitter: Double = 0.0,
        clockwise: Boolean = true,
        rng: Random = Random(Random.nextLong()),
        vx: Double = 0.0,
        vy: Double = 0.0,
        x: Double = config.widthPx * 0.5,
        y: Double = config.heightPx * 0.5,
        r: Double = 200.0,
        minR: Double = config.minRadius,
        centralMass: Double = config.centralMass,
        totalSatelliteMass: Double = config.totalSatelliteMass
    ): MutableList<Body> {
        val cx = x
        val cy = y
        val rMax = r
        val sats = (nTotal - 1).coerceAtLeast(0)
        val bodies = ArrayList<Body>(sats + 1)
        bodies += Body(cx, cy, vx, vy, centralMass)
        val mSat = if (sats > 0) totalSatelliteMass / sats else 0.0
        val rd = radialScale ?: (rMax / 3.0)
        val taperR = barTaperR ?: (rMax * 0.6)
        fun sampleExpRadius(): Double {
            val u = rng.nextDouble()
            val a = exp(-(rMax - minR) / rd)
            val t = 1 - u * (1 - a)
            return minR - rd * ln(t)
        }
        repeat(sats) {
            val radius = sampleExpRadius()
            val theta = rng.nextDouble() * 2.0 * Math.PI
            val taper = exp(- (radius / taperR) * (radius / taperR))
            val radiusPerturbed = radius * (1.0 + epsM2 * cos(2.0 * (theta - phi0)) * taper)
            val px = cx + radiusPerturbed * cos(theta)
            val py = cy + radiusPerturbed * sin(theta)
            bodies += Body(px, py, 0.0, 0.0, mSat)
        }
        data class RIdx(val index: Int, val radius: Double)
        val sorted = bodies.mapIndexed { i, b -> RIdx(i, hypot(b.x - cx, b.y - cy)) }.sortedBy { it.radius }
        var acc = 0.0
        val enclosed = DoubleArray(bodies.size)
        for (ri in sorted) {
            acc += bodies[ri.index].m
            enclosed[ri.index] = acc
        }
        for (i in 1 until bodies.size) {
            val body = bodies[i]
            val dx = body.x - cx
            val dy = body.y - cy
            val radius = max(1e-6, hypot(dx, dy))
            val vCirc = sqrt(config.g * enclosed[i] / radius)
            val v = vCirc * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * speedJitter)
            val tangent = if (clockwise) Pair(dy / radius, -dx / radius) else Pair(-dy / radius, dx / radius)
            var vx0 = tangent.first * v
            var vy0 = tangent.second * v
            if (radialJitter > 0.0) {
                val vr = (rng.nextDouble() - 0.5) * 2.0 * radialJitter * vCirc
                vx0 += (dx / radius) * vr
                vy0 += (dy / radius) * vr
            }
            body.vx = vx0 + vx
            body.vy = vy0 + vy
        }
        return bodies
    }
}
