import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

// =======================================================
//                       А Л Г О Р И Т М
//     (модель частиц, квадродерево Barnes–Hut, физика)
// =======================================================
data class Body(
    var x: Double, var y: Double,
    var vx: Double, var vy: Double,
    var m: Double
)

data class Quad(val cx: Double, val cy: Double, val h: Double) {
    fun contains(b: Body): Boolean =
        b.x >= cx - h && b.x < cx + h && b.y >= cy - h && b.y < cy + h
    fun child(which: Int): Quad {
        val hh = h / 2.0
        return when (which) {
            0 -> Quad(cx - hh, cy - hh, hh) // NW
            1 -> Quad(cx + hh, cy - hh, hh) // NE
            2 -> Quad(cx - hh, cy + hh, hh) // SW
            else -> Quad(cx + hh, cy + hh, hh) // SE
        }
    }
}

class BHTree(private val quad: Quad) {
    private var body: Body? = null
    private var children: Array<BHTree?>? = null
    var mass: Double = 0.0; private set
    var comX: Double = 0.0; private set
    var comY: Double = 0.0; private set

    private fun isLeaf(): Boolean = children == null

    fun insert(b: Body) {
        if (!quad.contains(b)) return
        if (body == null && isLeaf()) { body = b; return }
        if (isLeaf()) subdivide()
        body?.let { existing ->
            body = null
            insertIntoChild(existing)
        }
        insertIntoChild(b)
    }

    private fun insertIntoChild(b: Body) {
        if (quad.h < 1e-3) {
            b.x += (Random.nextDouble() - 0.5) * 1e-3
            b.y += (Random.nextDouble() - 0.5) * 1e-3
        }
        val ch = children!!
        val ix = if (b.x < quad.cx) 0 else 1
        val iy = if (b.y < quad.cy) 0 else 2
        ch[ix + iy]!!.insert(b)
    }

    private fun subdivide() {
        val ch = arrayOfNulls<BHTree>(4)
        for (i in 0 until 4) ch[i] = BHTree(quad.child(i))
        children = ch
    }

    fun computeMass() {
        if (isLeaf()) {
            body?.let {
                mass = it.m; comX = it.x; comY = it.y
            } ?: run { mass = 0.0; comX = quad.cx; comY = quad.cy }
        } else {
            var mSum = 0.0; var cx = 0.0; var cy = 0.0
            for (ch in children!!) {
                ch!!.computeMass()
                if (ch.mass > 0.0) {
                    mSum += ch.mass
                    cx += ch.comX * ch.mass
                    cy += ch.comY * ch.mass
                }
            }
            mass = mSum
            if (mSum > 0.0) { comX = cx / mSum; comY = cy / mSum } else { comX = quad.cx; comY = quad.cy }
        }
    }

    fun forceOn(b: Body, theta: Double): Pair<Double, Double> {
        if (mass == 0.0) return 0.0 to 0.0
        if (isLeaf()) {
            val single = body
            if (single == null || single === b) return 0.0 to 0.0
            return pointForce(b, comX, comY, mass)
        }
        val dx = comX - b.x
        val dy = comY - b.y
        val dist = sqrt(dx * dx + dy * dy + Config.SOFT2)
        val s = quad.h * 2.0
        return if (s / dist < theta) {
            pointForce(b, comX, comY, mass)
        } else {
            var fx = 0.0; var fy = 0.0
            for (ch in children!!) {
                val f = ch!!.forceOn(b, theta)
                fx += f.first; fy += f.second
            }
            fx to fy
        }
    }

    private fun pointForce(b: Body, px: Double, py: Double, m: Double): Pair<Double, Double> {
        val dx = px - b.x
        val dy = py - b.y
        val r2 = dx * dx + dy * dy + Config.SOFT2
        val r = sqrt(r2)
        val f = Config.G * b.m * m / r2
        return (f * dx / r) to (f * dy / r)
    }
}

class PhysicsEngine(initialBodies: MutableList<Body>) {
    private val cores = Runtime.getRuntime().availableProcessors()
    private var bodies: MutableList<Body> = initialBodies
    private var ax = DoubleArray(bodies.size)
    private var ay = DoubleArray(bodies.size)

    fun getBodies(): List<Body> = bodies

    fun resetBodies(newBodies: MutableList<Body>) {
        bodies = newBodies
        if (ax.size != bodies.size) { ax = DoubleArray(bodies.size); ay = DoubleArray(bodies.size) }
    }

    private fun buildTree(): BHTree {
        val half = max(Config.WIDTH_PX, Config.HEIGHT_PX) / 2.0 + 2.0
        val root = BHTree(Quad(Config.WIDTH_PX / 2.0, Config.HEIGHT_PX / 2.0, half))
        for (b in bodies) root.insert(b)
        root.computeMass()
        return root
    }

    private suspend fun computeAccelerations(root: BHTree) = coroutineScope {
        val n = bodies.size
        val chunk = max(256, n / (cores * 4).coerceAtLeast(1))
        for (start in 0 until n step chunk) {
            val end = min(n, start + chunk)
            launch(Dispatchers.Default) {
                var i = start
                while (i < end) {
                    val b = bodies[i]
                    val (fx, fy) = root.forceOn(b, Config.theta)
                    ax[i] = fx / b.m
                    ay[i] = fy / b.m
                    i++
                }
            }
        }
    }

    /** Один шаг Leapfrog (kick–drift–kick) */
    fun step() {
        // a(t)
        var root = buildTree()
        stepCalculations(root)

        // x(t+dt)
        for (b in bodies) {
            b.x += b.vx * Config.DT
            b.y += b.vy * Config.DT
        }

        // a(t+dt)
        root = buildTree()
        stepCalculations(root)
    }

    fun stepCalculations(root: BHTree) {
        runBlocking { computeAccelerations(root) }

        // v(t+dt/2)
        for (i in bodies.indices) {
            bodies[i].vx += ax[i] * (Config.DT * 0.5)
            bodies[i].vy += ay[i] * (Config.DT * 0.5)
        }
    }
}