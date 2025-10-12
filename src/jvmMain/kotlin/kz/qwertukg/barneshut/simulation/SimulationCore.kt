package kz.qwertukg.barneshut.simulation

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kz.qwertukg.barneshut.shared.BodySnapshot
import kz.qwertukg.barneshut.shared.SimulationConstants
import kz.qwertukg.barneshut.shared.SimulationSettings
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random
import java.util.concurrent.atomic.AtomicInteger

/** Состояние тела гравитационной системы Barnes–Hut. */
data class Body(
    var x: Double,
    var y: Double,
    var vx: Double,
    var vy: Double,
    var m: Double
)

/** Преобразование серверного тела в сериализуемый снимок. */
fun Body.toSnapshot(): BodySnapshot = BodySnapshot(x, y, vx, vy, m)

/** Аккумулятор для суммирования сил. */
private class Acc {
    var fx: Double = 0.0
    var fy: Double = 0.0
    fun reset() {
        fx = 0.0
        fy = 0.0
    }
}

/** Квадрат Barnes–Hut. */
private data class Quad(val cx: Double, val cy: Double, val h: Double) {
    fun contains(b: Body): Boolean =
        b.x >= cx - h && b.x < cx + h && b.y >= cy - h && b.y < cy + h

    fun child(which: Int): Quad {
        val hh = h / 2.0
        return when (which) {
            0 -> Quad(cx - hh, cy - hh, hh)
            1 -> Quad(cx + hh, cy - hh, hh)
            2 -> Quad(cx - hh, cy + hh, hh)
            else -> Quad(cx + hh, cy + hh, hh)
        }
    }
}

/** Узел квадродерева Barnes–Hut. */
private class BHTree(private val quad: Quad) {
    private var body: Body? = null
    private var children: Array<BHTree?>? = null
    var mass: Double = 0.0
        private set
    var comX: Double = 0.0
        private set
    var comY: Double = 0.0
        private set

    private fun isLeaf(): Boolean = children == null

    fun insert(b: Body) {
        if (!quad.contains(b)) return
        if (body == null && isLeaf()) {
            body = b
            return
        }
        if (isLeaf()) subdivide()
        body?.let { existing ->
            body = null
            insertIntoChild(existing)
        }
        insertIntoChild(b)
    }

    private fun insertIntoChild(b: Body) {
        if (quad.h < 1e-3) {
            val eps = 1e-3
            b.x += if ((b.x.toBits() and 1L) == 0L) +eps else -eps
            b.y += if ((b.y.toBits() and 1L) == 0L) -eps else +eps
        }
        val ch = children!!
        val ix = if (b.x < quad.cx) 0 else 1
        val iy = if (b.y < quad.cy) 0 else 2
        ch[ix + iy]!!.insert(b)
    }

    private fun subdivide() {
        children = arrayOf(
            BHTree(quad.child(0)),
            BHTree(quad.child(1)),
            BHTree(quad.child(2)),
            BHTree(quad.child(3))
        )
    }

    fun computeMass() {
        if (isLeaf()) {
            body?.let {
                mass = it.m
                comX = it.x
                comY = it.y
            } ?: run {
                mass = 0.0
                comX = quad.cx
                comY = quad.cy
            }
        } else {
            var mSum = 0.0
            var cx = 0.0
            var cy = 0.0
            val ch = children!!
            for (child in ch) {
                child!!.computeMass()
                if (child.mass > 0.0) {
                    mSum += child.mass
                    cx += child.comX * child.mass
                    cy += child.comY * child.mass
                }
            }
            mass = mSum
            if (mSum > 0.0) {
                comX = cx / mSum
                comY = cy / mSum
            } else {
                comX = quad.cx
                comY = quad.cy
            }
        }
    }

    private fun pointForceAcc(
        b: Body,
        px: Double,
        py: Double,
        m: Double,
        acc: Acc,
        gravity: Double,
        softeningSq: Double
    ) {
        val dx = px - b.x
        val dy = py - b.y
        val r2 = dx * dx + dy * dy + softeningSq
        val invR = 1.0 / sqrt(r2)
        val invR2 = 1.0 / r2
        val f = gravity * b.m * m * invR2
        acc.fx += f * dx * invR
        acc.fy += f * dy * invR
    }

    fun accumulateForce(b: Body, theta2: Double, acc: Acc, gravity: Double, softeningSq: Double) {
        if (mass == 0.0) return
        if (isLeaf()) {
            val single = body
            if (single == null || single === b) return
            pointForceAcc(b, comX, comY, mass, acc, gravity, softeningSq)
            return
        }
        val dx = comX - b.x
        val dy = comY - b.y
        val dist2 = dx * dx + dy * dy + softeningSq
        val s = quad.h * 2.0
        val s2 = s * s
        if (s2 < theta2 * dist2) {
            pointForceAcc(b, comX, comY, mass, acc, gravity, softeningSq)
        } else {
            val ch = children!!
            ch[0]!!.accumulateForce(b, theta2, acc, gravity, softeningSq)
            ch[1]!!.accumulateForce(b, theta2, acc, gravity, softeningSq)
            ch[2]!!.accumulateForce(b, theta2, acc, gravity, softeningSq)
            ch[3]!!.accumulateForce(b, theta2, acc, gravity, softeningSq)
        }
    }
}

/** Физический движок Barnes–Hut. */
class PhysicsEngine(
    initialBodies: MutableList<Body>,
    initialSettings: SimulationSettings
) {
    private val cores = Runtime.getRuntime().availableProcessors()
    private var bodies: MutableList<Body> = initialBodies
    private var ax = DoubleArray(bodies.size)
    private var ay = DoubleArray(bodies.size)
    private var settings: SimulationSettings = initialSettings

    fun getBodies(): List<Body> = bodies

    fun resetBodies(newBodies: MutableList<Body>) {
        bodies = newBodies
        if (ax.size != bodies.size) {
            ax = DoubleArray(bodies.size)
            ay = DoubleArray(bodies.size)
        }
    }

    fun updateSettings(newSettings: SimulationSettings) {
        settings = newSettings
    }

    private fun buildTree(): BHTree {
        val half = max(settings.widthPx, settings.heightPx) / 2.0 + 2.0
        val root = BHTree(Quad(settings.widthPx / 2.0, settings.heightPx / 2.0, half))
        for (b in bodies) root.insert(b)
        root.computeMass()
        return root
    }

    private suspend fun computeAccelerations(root: BHTree) = coroutineScope {
        val bs = bodies
        val n = bs.size
        val workers = min(cores, n.coerceAtLeast(1))
        val theta2 = settings.theta * settings.theta
        val gravity = settings.gravity
        val softeningSq = settings.softeningSquared
        val next = AtomicInteger(0)

        repeat(workers) {
            launch(Dispatchers.Default) {
                val acc = Acc()
                while (true) {
                    val i = next.getAndIncrement()
                    if (i >= n) break
                    val b = bs[i]
                    acc.reset()
                    root.accumulateForce(b, theta2, acc, gravity, softeningSq)
                    ax[i] = acc.fx / b.m
                    ay[i] = acc.fy / b.m
                }
            }
        }
    }

    fun step() {
        if (bodies.isEmpty()) return
        var root = buildTree()
        runBlocking { computeAccelerations(root) }

        val dt = settings.timeStep
        val dtHalf = dt * 0.5
        for (i in bodies.indices) {
            bodies[i].vx += ax[i] * dtHalf
            bodies[i].vy += ay[i] * dtHalf
        }

        for (b in bodies) {
            b.x += b.vx * dt
            b.y += b.vy * dt
        }

        root = buildTree()
        runBlocking { computeAccelerations(root) }

        for (i in bodies.indices) {
            bodies[i].vx += ax[i] * dtHalf
            bodies[i].vy += ay[i] * dtHalf
        }
    }
}

/** Генерация стартовых конфигураций тел. */
object BodyFactory {
    fun makeKeplerDisk(
        settings: SimulationSettings,
        nTotal: Int = settings.bodiesPerDisk,
        clockwise: Boolean = true,
        radialJitter: Double = 0.03,
        speedJitter: Double = 0.01,
        rng: Random = Random(3),
        vx: Double = 0.0,
        vy: Double = 0.0,
        x: Double = settings.widthPx * 0.5,
        y: Double = settings.heightPx * 0.5,
        r: Double = min(settings.widthPx, settings.heightPx) * 0.38
    ): MutableList<Body> {
        val cx = x
        val cy = y
        val rMax = r
        val sats = (nTotal - 1).coerceAtLeast(0)
        val bodies = ArrayList<Body>(sats + 1)

        bodies += Body(cx, cy, vx, vy, SimulationConstants.CENTRAL_MASS)

        val mSat = if (sats > 0) SimulationConstants.TOTAL_SATELLITE_MASS / sats else 0.0

        repeat(sats) {
            val u = rng.nextDouble()
            val rr = sqrt(
                u * (rMax * rMax - SimulationConstants.MIN_RADIUS * SimulationConstants.MIN_RADIUS) +
                    SimulationConstants.MIN_RADIUS * SimulationConstants.MIN_RADIUS
            )
            val rJ = rr * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * radialJitter)
            val ang = rng.nextDouble() * 2.0 * Math.PI
            val bx = cx + rJ * cos(ang)
            val by = cy + rJ * sin(ang)
            bodies += Body(bx, by, 0.0, 0.0, mSat)
        }

        data class RIdx(val i: Int, val r: Double)
        val sorted = bodies.mapIndexed { i, b -> RIdx(i, hypot(b.x - cx, b.y - cy)) }.sortedBy { it.r }
        var accMass = 0.0
        val mEnclosed = DoubleArray(bodies.size)
        for (ri in sorted) {
            accMass += bodies[ri.i].m
            mEnclosed[ri.i] = accMass
        }

        for (i in 1 until bodies.size) {
            val b = bodies[i]
            val dx = b.x - cx
            val dy = b.y - cy
            val radius = max(1e-6, hypot(dx, dy))
            val vCirc = sqrt(settings.gravity * mEnclosed[i] / radius)
            val v = vCirc * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * speedJitter)
            val (tx, ty) = if (clockwise) (dy / radius) to (-dx / radius) else (-dy / radius) to (dx / radius)
            b.vx = tx * v + vx
            b.vy = ty * v + vy
        }

        return bodies
    }

    fun defaultBodies(settings: SimulationSettings): MutableList<Body> {
        val drift = 70.0
        val first = makeKeplerDisk(
            settings = settings,
            nTotal = settings.bodiesPerDisk,
            vx = drift,
            vy = 0.0,
            x = settings.widthPx * 0.5,
            y = settings.heightPx * 0.4,
            r = settings.diskRadius,
            clockwise = true
        )
        val second = makeKeplerDisk(
            settings = settings,
            nTotal = settings.bodiesPerDisk,
            vx = -drift,
            vy = 0.0,
            x = settings.widthPx * 0.5,
            y = settings.heightPx * 0.6,
            r = settings.diskRadius,
            clockwise = true
        )
        return (first + second).toMutableList()
    }
}
