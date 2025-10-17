import java.awt.BasicStroke
import java.awt.Color
import java.awt.Dimension
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.Point
import java.awt.RenderingHints
import java.awt.Toolkit
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import java.awt.event.MouseWheelEvent
import javax.swing.AbstractAction
import javax.swing.JPanel
import javax.swing.KeyStroke
import javax.swing.Timer
import kotlin.random.Random
import kotlin.system.exitProcess

/**
 * Панель Swing, отвечающая за визуализацию и управление симуляцией Barnes–Hut.
 *
 * Возможности:
 * - ЛКМ drag’n’drop: создать кеплеровский диск; длина и направление линии → начальная (vx, vy).
 * - Средняя кнопка: очистить сцену.
 * - Колесо мыши: зум (×1 … ×10) вокруг курсора.
 * - Стрелки ← → ↑ ↓: панорамирование (смещение окна просмотра) в мировых координатах.
 * - Клавиши: см. HUD/биндинги в setupKeys().
 *
 * Проекция «мир → экран»:
 *      screenX = (worldX - viewX) * zoom
 *      screenY = (worldY - viewY) * zoom
 *      worldX  = viewX + screenX / zoom
 *      worldY  = viewY + screenY / zoom
 */
class NBodyPanel : JPanel() {

    /** Точка начала перетаскивания мышью (в экранных координатах). */
    private var dragStart: Point? = null

    /** Текущее положение курсора во время перетаскивания (в экранных координатах). */
    private var dragCurrent: Point? = null

    /** Перевод пикселей перетаскивания в скорость в мировых единицах (px/сек). */
    private val VEL_PER_PIXEL = 1   // 1 px перетаскивания = 1 ед. скорости

    /** Базовая скорость по X/Y для кликовых дисков. */
    private val initialVX = 0.0
    private val initialVY = 0.0

    /** Показывать ли границы квадродерева. */
    private var showTree = false

    // ---------- Viewport (проекция «мир → экран») ----------
    /** Текущий масштаб (1.0 … 10.0). */
    private var zoom = 1.0
    private val zoomMin = 1.0
    private val zoomMax = 10.0
    private val zoomStep = 1.1 // множитель на один «щёлчок» колеса

    /** Смещение окна просмотра (верх-левый угол) в мировых координатах. */
    private var viewX = 0.0
    private var viewY = 0.0

    /** Шаг панорамирования в пикселях экрана на одно нажатие стрелки (переводится в мир через деление на zoom). */
    private val panStepScreen = 10.0

    /** Преобразование мир→экран. */
    private fun worldToScreenX(wx: Double): Int = ((wx - viewX) * zoom).toInt()
    private fun worldToScreenY(wy: Double): Int = ((wy - viewY) * zoom).toInt()

    /** Преобразование экран→мир. */
    private fun screenToWorldX(sx: Double): Double = viewX + sx / zoom
    private fun screenToWorldY(sy: Double): Double = viewY + sy / zoom

    private var fps = 0
    private var frames = 0
    private var lastSec = System.currentTimeMillis()

    /**
     * Сформировать стартовый набор тел: два кеплеровских диска с противоположным дрейфом.
     * Следуем рекомендациям DAML по симметричной постановке задачи.
     */
    fun defaultBodies(): MutableList<Body> {
        val disc1 = BodyFactory.makeGalaxyDisk(
            10_000,
            r = 300.0,
            centralMass = 50_000.0,
            totalSatelliteMass = 5_000.0
        )
        val disc2 = BodyFactory.makeGalaxyDisk(
            2_500,
            y = Config.HEIGHT_PX * 0.2,
            vx = -50.0,
            r = 100.0,
            centralMass = 5_000.0,
            totalSatelliteMass = 500.0
        )

        return (disc1 + disc2).toMutableList()
    }

    /** Движок симуляции для текущего множества тел. */
    private var engine = PhysicsEngine(defaultBodies())

    /** Периодический таймер перерисовки и шагов симуляции. */
    private val timer = Timer(1) { tick() } // ~60 FPS

    /** Флаг постановки симуляции на паузу. */
    private var paused = false

    init {
        preferredSize = Dimension(Config.WIDTH_PX, Config.HEIGHT_PX)
        background = Color.BLACK
        isFocusable = true
        setupKeys()
        setupMouse()
        timer.start()
    }

    /** Ограничить камеру так, чтобы видимая область полностью находилась в пределах базового мира. */
    private fun clampView() {
        val visibleW = width.toDouble() / zoom
        val visibleH = height.toDouble() / zoom
        val maxX = (Config.WIDTH_PX - visibleW).coerceAtLeast(0.0)
        val maxY = (Config.HEIGHT_PX - visibleH).coerceAtLeast(0.0)
        viewX = viewX.coerceIn(0.0, maxX)
        viewY = viewY.coerceIn(0.0, maxY)
    }

    /** Настроить обработку мыши (клик/drag и колесо для зума). */
    private fun setupMouse() {
        val mouse = object : MouseAdapter() {
            override fun mousePressed(e: MouseEvent) {
                if (e.button == MouseEvent.BUTTON1 || e.button == MouseEvent.BUTTON3) {
                    dragStart = e.point
                    dragCurrent = e.point
                    when (e.button) {
                        MouseEvent.BUTTON1 -> uiR = Config.R
                        MouseEvent.BUTTON3 -> uiR = Config.MIN_R
                    }
                    repaint()
                }
                if (e.button == MouseEvent.BUTTON2) {
                    engine.resetBodies(mutableListOf())
                    repaint()
                }
            }
            override fun mouseDragged(e: MouseEvent) {
                if (dragStart != null) {
                    dragCurrent = e.point
                    repaint()
                }
            }
            override fun mouseReleased(e: MouseEvent) {
                if ((e.button == MouseEvent.BUTTON1 || e.button == MouseEvent.BUTTON3) && dragStart != null) {
                    val start = dragStart!!
                    val end = e.point ?: start
                    val dxScreen = (end.x - start.x).toDouble()
                    val dyScreen = (end.y - start.y).toDouble()

                    // Скорости в мире → делим на zoom
                    val vx = (dxScreen / zoom) * VEL_PER_PIXEL
                    val vy = (dyScreen / zoom) * VEL_PER_PIXEL

                    // Центр диска — мировые координаты
                    val wx = screenToWorldX(start.x.toDouble())
                    val wy = screenToWorldY(start.y.toDouble())

                    when (e.button) {
                        MouseEvent.BUTTON1 -> addGalaxyDiskAt(wx, wy, Config.R, Config.N, vx, vy)
                        MouseEvent.BUTTON3 -> addGalaxyDiskAt(wx, wy, Config.MIN_R, 0, vx, vy)
                    }

                    dragStart = null
                    dragCurrent = null
                    repaint()
                }
            }

            override fun mouseWheelMoved(e: MouseWheelEvent) {
                val sx = e.x.toDouble()
                val sy = e.y.toDouble()

                val wx = screenToWorldX(sx)
                val wy = screenToWorldY(sy)

                val factor = if (e.preciseWheelRotation < 0) zoomStep else 1.0 / zoomStep
                val newZoom = (zoom * factor).coerceIn(zoomMin, zoomMax)

                if (newZoom != zoom) {
                    viewX = wx - sx / newZoom
                    viewY = wy - sy / newZoom
                    zoom = newZoom
                    clampView()
                }
            }

//            override fun mouseClicked(e: MouseEvent) {
//                if (e.button == MouseEvent.BUTTON3) {
//                    val wx = screenToWorldX(e.x.toDouble())
//                    val wy = screenToWorldY(e.y.toDouble())
//                    addGalaxyDiskAt(wx, wy, 0.0, 0)
//                }
//            }
        }
        addMouseListener(mouse)
        addMouseMotionListener(mouse) // для drag
        addMouseWheelListener(mouse)  // колесо — зум
    }

    /**
     * Добавить новый кеплеровский диск по указанным параметрам.
     * @param x координата центра по X (мир).
     * @param y координата центра по Y (мир).
     * @param r радиус диска (мир).
     * @param n число тел.
     * @param vx добавочная скорость по X (мир).
     * @param vy добавочная скорость по Y (мир).
     */
    private fun addKeplerDiskAt(x: Double, y: Double, r: Double, n: Int, vx: Double = initialVY, vy: Double = initialVX) {
        val newDisk = BodyFactory.makeKeplerDisk(
            nTotal = n, vx = vx, vy = vy, x = x, y = y, r = r
        )
        val merged = (engine.getBodies() + newDisk).toMutableList()
        engine.resetBodies(merged)
    }

    private fun addGalaxyDiskAt(x: Double, y: Double, r: Double, n: Int, vx: Double = initialVY, vy: Double = initialVX) {
        val newDisk = BodyFactory.makeGalaxyDisk(
            nTotal = n, vx = vx, vy = vy, x = x, y = y, r = r
        )
        val merged = (engine.getBodies() + newDisk).toMutableList()
        engine.resetBodies(merged)
    }

    /** Настроить горячие клавиши для управления симуляцией (включая панорамирование стрелками). */
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
        bind("Z") { Config.theta = (Config.theta - 0.05).coerceAtLeast(0.2) }
        bind("X") { Config.theta = (Config.theta + 0.05).coerceAtMost(1.6) }

        bind("A") { Config.N = (Config.N - 100).coerceAtLeast(1000) }
        bind("S") { Config.N = (Config.N + 100).coerceAtMost(10000) }

        bind("Q") { Config.R = (Config.R - 10.0).coerceAtLeast(100.0) }
        bind("W") { Config.R = (Config.R + 10.0).coerceAtMost(500.0) }

        bind("O") { Config.DT = (Config.DT - 0.001).coerceAtLeast(-0.05) }
        bind("P") { Config.DT = (Config.DT + 0.001).coerceAtMost(0.05) }

        bind("K") { Config.G = (Config.G - 1.0).coerceAtLeast(0.0) }
        bind("L") { Config.G = (Config.G + 1.0).coerceAtMost(100.0) }

        bind("R") { engine.resetBodies(defaultBodies()) }
        bind("ESCAPE") { exitProcess(0) }

        // Показать/скрыть границы квадродерева
        bind("D") { showTree = !showTree; repaint() }

        // -------- Панорамирование стрелками (смещение окна просмотра) --------
        // Шаг в мире соответствует фиксированному пиксельному шагу на экране
        fun pan(dxScreen: Double, dyScreen: Double) {
            val dxWorld = dxScreen / zoom
            val dyWorld = dyScreen / zoom
            viewX += dxWorld
            viewY += dyWorld
            clampView()
        }
        bind("LEFT")  { pan(-panStepScreen, 0.0) }
        bind("RIGHT") { pan(+panStepScreen, 0.0) }
        bind("UP")    { pan(0.0, -panStepScreen) }
        bind("DOWN")  { pan(0.0, +panStepScreen) }

        bind("C") {
            val cloud = BodyFactory.makeUniformRandom(n = 5_000, m = 0.5)
            val bodies = (engine.getBodies() + cloud).toMutableList()
            engine.resetBodies(bodies)
        }
    }

    /** Один кадр визуализации и, при необходимости, шаг симуляции. */
    private fun tick() {
        if (!paused) engine.step()
        repaint()
    }

    /** Отрисовать все тела, превью драга, границы квадродерева и HUD. */
    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        val g2 = g as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)

        // Точки тел (мир → экран)
        for (b in engine.getBodies()) {
            g2.color = if (b.m >= 1000) Color.BLACK else Color.WHITE
            val sx = worldToScreenX(b.x)
            val sy = worldToScreenY(b.y)
            if (sx in 0 until width && sy in 0 until height) g2.drawLine(sx, sy, sx, sy)
        }

        // Линия drag preview (экранные координаты)
        if (dragStart != null && dragCurrent != null) {
            val sx = dragStart!!.x; val sy = dragStart!!.y
            val ex = dragCurrent!!.x; val ey = dragCurrent!!.y
            val oldStroke = g2.stroke
            g2.color = Color(0, 255, 0, 255)
            g2.stroke = BasicStroke(
                1.0f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                10f, floatArrayOf(6f, 6f), 0f
            )
            g2.drawLine(sx, sy, ex, ey)
            // Превью радиуса диска — r (мир) → r*zoom (экран)
            val rScreen = (uiR * zoom).toInt()
            g2.drawArc(sx - rScreen, sy - rScreen, rScreen * 2, rScreen * 2, 0, 360)
            g2.stroke = oldStroke
        }

        // Границы квадродерева (мир → экран)
        if (showTree) {
            val oldColor = g2.color
            val oldStroke = g2.stroke
            g2.color = Color(0, 255, 0, 180)
            g2.stroke = BasicStroke(1f)

            val tree = engine.getTreeForDebug()
            tree.visitQuads { q ->
                val x = worldToScreenX(q.cx - q.h)
                val y = worldToScreenY(q.cy - q.h)
                val s = (q.h * 2.0 * zoom).toInt()
                g2.drawLine(x, y, x, y + s)
                g2.drawLine(x, y, x + s, y)
            }

            g2.color = oldColor
            g2.stroke = oldStroke
        }

        // HUD
        g2.color = Color(0, 255, 0)
        g2.drawString("SPACE — pause | R — reset scene | MBL DRAG'N'DROP — add galaxy disk | ARROWS — cam movement | ESCAPE — exit", 10, 20)
        g2.drawString("Disk radius [Q/W] = ${Config.R}", 10, 60)
        g2.drawString("Bodies count [A/S] = ${Config.N}", 10, 80)
        g2.drawString("Theta [Z/X] = ${Config.theta}", 10, 100)
        g2.drawString("Delta time [O/P] = ${Config.DT}", 10, 120)
        g2.drawString("Gravity [K/L] = ${Config.G}", 10, 140)
        g2.drawString("Debug mode [D] = $showTree", 10, 160)
        g2.drawString("Zoom [WHEEL] = $zoom", 10, 180)
        g2.drawString("Bodies count = ${engine.getBodies().size}", 10, 200)
        g2.drawString("Softening = ${Config.SOFTENING}", 10, 220)
        g2.drawString("Create bodies cloud [C]", 10, 240)
        g2.drawString("Create black hole [MBR DRAG'N'DROP]", 10, 260)

        frames++
        val now = System.currentTimeMillis()
        if (now - lastSec >= 1000) {
            fps = frames
            frames = 0
            lastSec = now
        }
        g2.drawString("FPS: $fps", 10, 280) // координаты подгони по вкусу

        Toolkit.getDefaultToolkit().sync()
    }

    private var uiR = Config.R
}