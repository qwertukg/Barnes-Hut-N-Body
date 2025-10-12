// Main.kt
import java.awt.*
import javax.swing.*
import kotlin.math.*
import kotlin.random.Random
import kotlinx.coroutines.*

// ---------------------- ПАРАМЕТРЫ ----------------------
private const val WIDTH_PX = 1200
private const val HEIGHT_PX = 800

private const val G = 80.0                 // грав. константа в "пиксельных" единицах
private const val THETA0 = 0.35            // параметр приближения Барнса–Хатта
private const val DT = 0.015               // шаг интегрирования (сек)
private const val SOFTENING = 1.0          // смягчение сингулярности
private val SOFT2 = SOFTENING * SOFTENING
private const val BODIES_COUNT = 2000

// Начальная сцена — орбитальная система
private const val CENTRAL_MASS = 50_000.0
private const val SATELLITE_MASS = 0.5
private const val MIN_R = 24.0

// ---------------------- МОДЕЛЬ ----------------------
data class Body(
    var x: Double, var y: Double,
    var vx: Double, var vy: Double,
    var m: Double
)

private data class Quad(val cx: Double, val cy: Double, val h: Double) {
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

private class BHTree(private val quad: Quad) {
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
        if (quad.h < 1e-3) { // защита от деградации при совпадающих координатах
            b.x += (Random.nextDouble() - 0.5) * 1e-3
            b.y += (Random.nextDouble() - 0.5) * 1e-3
        }
        val ch = children!!
        val ix = if (b.x < quad.cx) 0 else 1
        val iy = if (b.y < quad.cy) 0 else 2
        val which = ix + iy
        ch[which]?.insert(b)
    }

    private fun subdivide() {
        val ch = arrayOfNulls<BHTree>(4)
        for (i in 0 until 4) ch[i] = BHTree(quad.child(i))
        children = ch
    }

    fun computeMass() {
        if (isLeaf()) {
            body?.let {
                mass = it.m
                comX = it.x
                comY = it.y
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
        val dist = sqrt(dx * dx + dy * dy + SOFT2)
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
        val r2 = dx * dx + dy * dy + SOFT2
        val r = sqrt(r2)
        val f = G * b.m * m / r2
        return (f * dx / r) to (f * dy / r)
    }
}

// ---------------------- УТИЛИТЫ СЦЕН ----------------------
/** Сгенерировать систему: массивное центральное тело + диск спутников.
 *  Скорости спутников — по круговой формуле v(r)=sqrt(G*M_enclosed(r)/r). */
private fun makeKeplerDisk(
    cx: Double,
    cy: Double,
    centralMass: Double,
    satelliteCount: Int,
    minR: Double,
    maxR: Double,
    clockwise: Boolean = true,
    radialJitter: Double = 0.03,
    speedJitter: Double = 0.01,
    totalSatelliteMass: Double = 1000.0, // ← фиксируем общую массу диска
    rng: Random = Random(2)
): MutableList<Body> {
    require(satelliteCount >= 0)
    val bodies = ArrayList<Body>(satelliteCount + 1)

    // центр
    bodies += Body(cx, cy, 0.0, 0.0, centralMass)

    // масса одного спутника так, чтобы сумма была постоянной при любом N
    val mSat = if (satelliteCount > 0) totalSatelliteMass / satelliteCount else 0.0

    // позиции без скоростей
    repeat(satelliteCount) {
        val u = rng.nextDouble()
        val rr = sqrt(u * (maxR*maxR - minR*minR) + minR*minR)
        val rJ = rr * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * radialJitter)
        val ang = rng.nextDouble() * 2.0 * Math.PI
        val x = cx + rJ * cos(ang)
        val y = cy + rJ * sin(ang)
        bodies += Body(x, y, 0.0, 0.0, mSat)
    }

    // считаем M_enclosed(r) и выставляем круговые скорости
    data class RIdx(val i:Int, val r:Double)
    val sorted = bodies.mapIndexed { i,b -> RIdx(i, hypot(b.x-cx, b.y-cy)) }.sortedBy { it.r }
    var acc = 0.0
    val Menc = DoubleArray(bodies.size)
    for (ri in sorted) { acc += bodies[ri.i].m; Menc[ri.i] = acc }

    for (i in 1 until bodies.size) { // i=0 — центр
        val b = bodies[i]
        val dx = b.x - cx; val dy = b.y - cy
        val r = max(1e-6, hypot(dx, dy))
        val vCirc = sqrt(G * Menc[i] / r)
        val v = vCirc * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * speedJitter)
        val (tx, ty) = if (clockwise) (dy/r) to (-dx/r) else (-dy/r) to (dx/r)
        b.vx = tx * v
        b.vy = ty * v
    }
    return bodies
}


// ---------------------- UI / СИМУЛЯТОР ----------------------
private class BHPanel : JPanel() {
    private var bodies: MutableList<Body> = mutableListOf()
    private var theta = THETA0
    private var paused = false
    private val timer = Timer(16) { tick() } // ~60 FPS
    private val cores = Runtime.getRuntime().availableProcessors()

    // Буферы ускорений
    private var ax = DoubleArray(0)
    private var ay = DoubleArray(0)

    init {
        preferredSize = Dimension(WIDTH_PX, HEIGHT_PX)
        background = Color.BLACK
        isFocusable = true
        setupKeys()
        reset(BODIES_COUNT)
        timer.start()
    }

    private fun setupKeys() {
        fun bind(key: String, action: () -> Unit) {
            val im = getInputMap(WHEN_IN_FOCUSED_WINDOW)
            val am = actionMap
            im.put(KeyStroke.getKeyStroke(key), key)
            am.put(key, object : AbstractAction() {
                override fun actionPerformed(e: java.awt.event.ActionEvent?) = action()
            })
        }
        bind("SPACE") { paused = !paused }
        bind("R") { reset(bodies.size) }
        bind("X") { theta = (theta + 0.05).coerceAtMost(1.6) }
        bind("Z") { theta = (theta - 0.05).coerceAtLeast(0.2) }
        bind("OPEN_BRACKET") { reset((bodies.size * 0.9).toInt().coerceAtLeast(50)) }
        bind("CLOSE_BRACKET") { reset((bodies.size * 1.1).toInt().coerceAtMost(20000)) }
    }

    private fun ensureBuffers(n: Int) {
        if (ax.size != n) { ax = DoubleArray(n); ay = DoubleArray(n) }
    }

    private fun reset(n: Int) {
        val cx = WIDTH_PX * 0.5
        val cy = HEIGHT_PX * 0.5
        val rMax = min(WIDTH_PX, HEIGHT_PX) * 0.38

        // Орбитальная сцена: центр + диск
        bodies = makeKeplerDisk(
            cx = cx, cy = cy,
            centralMass = CENTRAL_MASS,
            satelliteCount = (n - 1).coerceAtLeast(0),
            minR = MIN_R, maxR = rMax,
            clockwise = true,
            radialJitter = 0.03,
            speedJitter = 0.01,
            rng = Random(3)
        )
        ensureBuffers(bodies.size)
    }

    // Построить дерево
    private fun buildTree(): BHTree {
        val rootHalf = max(WIDTH_PX, HEIGHT_PX) / 2.0 + 2.0
        val root = BHTree(Quad(WIDTH_PX / 2.0, HEIGHT_PX / 2.0, rootHalf))
        for (b in bodies) root.insert(b)
        root.computeMass()
        return root
    }

    // Параллельный расчёт ускорений в ax/ay
    private suspend fun computeAccelerations(root: BHTree) = coroutineScope {
        val n = bodies.size
        val chunk = max(256, n / (cores * 4).coerceAtLeast(1))
        for (start in 0 until n step chunk) {
            val end = min(n, start + chunk)
            launch(Dispatchers.Default) {
                var i = start
                while (i < end) {
                    val b = bodies[i]
                    val (fx, fy) = root.forceOn(b, theta)
                    ax[i] = fx / b.m
                    ay[i] = fy / b.m
                    i++
                }
            }
        }
    }

    private fun tick() {
        if (paused) { repaint(); return }

        // ---- Leapfrog (kick–drift–kick) ----
        // 1) a(t)
        var root = buildTree()
        runBlocking { computeAccelerations(root) }

        // 2) Kick: v(t+dt/2)
        for (i in bodies.indices) {
            bodies[i].vx += ax[i] * (DT * 0.5)
            bodies[i].vy += ay[i] * (DT * 0.5)
        }

        // 3) Drift: x(t+dt)
        for (b in bodies) {
            b.x += b.vx * DT
            b.y += b.vy * DT
            // без обёрток/отражений — держим орбиты в глубине окна подбором maxR
        }

        // 4) a(t+dt)
        root = buildTree()
        runBlocking { computeAccelerations(root) }

        // 5) Kick: v(t+dt)
        for (i in bodies.indices) {
            bodies[i].vx += ax[i] * (DT * 0.5)
            bodies[i].vy += ay[i] * (DT * 0.5)
        }

        repaint()
    }

    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        val g2 = g as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
        g2.color = Color.WHITE
        for (b in bodies) {
            val ix = b.x.toInt()
            val iy = b.y.toInt()
            if (ix in 0 until width && iy in 0 until height) g2.drawLine(ix, iy, ix, iy) // 1-пиксельные точки
        }
        g2.color = Color(255, 255, 255, 200)
        g2.drawString("N=${bodies.size}  θ=%.2f  dt=%.3f  G=%.1f".format(theta, DT, G), 10, 20)
        g2.drawString("SPACE=pause | R=reset | Z/X θ± | [ ] N±".trim(), 10, 36)
        Toolkit.getDefaultToolkit().sync()
    }
}

// ---------------------- ENTRY ----------------------
fun main() {
    SwingUtilities.invokeLater {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName())
        JFrame("Barnes–Hut N-Body • Orbits • Kotlin + Coroutines").apply {
            defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            contentPane.add(BHPanel())
            pack()
            setLocationRelativeTo(null)
            isVisible = true
        }
    }
}
