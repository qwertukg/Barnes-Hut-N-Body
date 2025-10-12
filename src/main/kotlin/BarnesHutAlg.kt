import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

// =======================================================
//                       А Л Г О Р И Т М
//     (модель частиц, квадродерево Barnes–Hut, физика)
//   — применены оптимизации без изменения параметров —
//   1) нет Pair в рекурсии (аккумулятор Acc)
//   2) критерий Barnes–Hut без sqrt (в квадратах)
//   3) фиксированный пул корутин с AtomicInteger
//   4) микрооптимизации в математике и subdivide()
// =======================================================

data class Body(
    var x: Double, var y: Double,
    var vx: Double, var vy: Double,
    var m: Double
)

/** Небольшой аккумулятор сил (переиспользуется внутри воркера). */
class Acc {
    var fx = 0.0
    var fy = 0.0
    fun reset() { fx = 0.0; fy = 0.0 }
}

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
        // защита от деградации при совпадающих координатах — без Random в горячем пути
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
            BHTree(quad.child(3)),
        )
    }

    fun computeMass() {
        if (isLeaf()) {
            body?.let {
                mass = it.m; comX = it.x; comY = it.y
            } ?: run { mass = 0.0; comX = quad.cx; comY = quad.cy }
        } else {
            var mSum = 0.0; var cx = 0.0; var cy = 0.0
            val ch = children!!
            ch[0]!!.computeMass(); if (ch[0]!!.mass > 0.0) { mSum += ch[0]!!.mass; cx += ch[0]!!.comX * ch[0]!!.mass; cy += ch[0]!!.comY * ch[0]!!.mass }
            ch[1]!!.computeMass(); if (ch[1]!!.mass > 0.0) { mSum += ch[1]!!.mass; cx += ch[1]!!.comX * ch[1]!!.mass; cy += ch[1]!!.comY * ch[1]!!.mass }
            ch[2]!!.computeMass(); if (ch[2]!!.mass > 0.0) { mSum += ch[2]!!.mass; cx += ch[2]!!.comX * ch[2]!!.mass; cy += ch[2]!!.comY * ch[2]!!.mass }
            ch[3]!!.computeMass(); if (ch[3]!!.mass > 0.0) { mSum += ch[3]!!.mass; cx += ch[3]!!.comX * ch[3]!!.mass; cy += ch[3]!!.comY * ch[3]!!.mass }
            mass = mSum
            if (mSum > 0.0) { comX = cx / mSum; comY = cy / mSum } else { comX = quad.cx; comY = quad.cy }
        }
    }

    // --- Быстрый расчёт точечной силы: минимум делений, без аллокаций ---
    private fun pointForceAcc(b: Body, px: Double, py: Double, m: Double, acc: Acc) {
        val dx = px - b.x
        val dy = py - b.y
        val r2 = dx*dx + dy*dy + Config.SOFT2
        val invR = 1.0 / sqrt(r2)
        val invR2 = 1.0 / r2
        val f = Config.G * b.m * m * invR2
        acc.fx += f * dx * invR
        acc.fy += f * dy * invR
    }

    /** Накопить силу на b в acc. Критерий Барнса–Хатта в квадратах: s^2 < θ^2 * dist^2 */
    fun accumulateForce(b: Body, theta2: Double, acc: Acc) {
        if (mass == 0.0) return
        if (isLeaf()) {
            val single = body
            if (single == null || single === b) return
            pointForceAcc(b, comX, comY, mass, acc)
            return
        }
        val dx = comX - b.x
        val dy = comY - b.y
        val dist2 = dx*dx + dy*dy + Config.SOFT2
        val s2 = (quad.h * 2.0).let { it * it }

        if (s2 < theta2 * dist2) {
            pointForceAcc(b, comX, comY, mass, acc)
        } else {
            val ch = children!!
            ch[0]!!.accumulateForce(b, theta2, acc)
            ch[1]!!.accumulateForce(b, theta2, acc)
            ch[2]!!.accumulateForce(b, theta2, acc)
            ch[3]!!.accumulateForce(b, theta2, acc)
        }
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
        val bs = bodies
        for (b in bs) root.insert(b)
        root.computeMass()
        return root
    }

    /** Параллельный расчёт ускорений (фиксированное число воркеров). */
    private suspend fun computeAccelerations(root: BHTree) = coroutineScope {
        val bs = bodies
        val n = bs.size
        val workers = min(cores, n.coerceAtLeast(1))
        val theta2 = Config.theta * Config.theta
        val next = AtomicInteger(0)

        repeat(workers) {
            launch(Dispatchers.Default) {
                val acc = Acc() // переиспользуем в рамках воркера
                while (true) {
                    val i = next.getAndIncrement()
                    if (i >= n) break
                    val b = bs[i]
                    acc.reset()
                    root.accumulateForce(b, theta2, acc)
                    ax[i] = acc.fx / b.m
                    ay[i] = acc.fy / b.m
                }
            }
        }
    }

    /** Один шаг Leapfrog (kick–drift–kick). */
    fun step() {
        // a(t)
        var root = buildTree()
        runBlocking { computeAccelerations(root) }

        // v(t+dt/2)
        val bs = bodies
        val dtHalf = Config.DT * 0.5
        for (i in bs.indices) {
            bs[i].vx += ax[i] * dtHalf
            bs[i].vy += ay[i] * dtHalf
        }

        // x(t+dt)
        for (b in bs) {
            b.x += b.vx * Config.DT
            b.y += b.vy * Config.DT
        }

        // a(t+dt)
        root = buildTree()
        runBlocking { computeAccelerations(root) }

        // v(t+dt)
        for (i in bs.indices) {
            bs[i].vx += ax[i] * dtHalf
            bs[i].vy += ay[i] * dtHalf
        }
    }
}
