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
import kotlin.system.exitProcess

// =======================================================
//                    В И З У А Л И З А Ц И Я
//            (отображение, горячие клавиши, HUD)
//     ЛКМ — создать новый Kepler-диск в точке клика
// =======================================================
class NBodyPanel : JPanel() {


    // drag state
    private var dragStart: Point? = null
    private var dragCurrent: Point? = null
    private val VEL_PER_PIXEL = 1   // 1 px перетаскивания = 0.05 ед. скорости (подбери под себя)

    // Базовые настройки для создаваемых кликом дисков
    private val clickDiskRadius = 100.0
    private val initialVX = 0.0
    private val initialVY = 0.0

    // Стартовая сцена: один статичный диск по центру окна
    fun defaultBodies(): MutableList<Body> {
        val s = 70.0
        val disc1 = BodyFactory.makeKeplerDisk(
            nTotal = 2000,
            vx = s,
            vy = 0.0,
            x = Config.WIDTH_PX * 0.5,
            y = Config.HEIGHT_PX * 0.4,
            r = 100.0
        )
        val disc2 = BodyFactory.makeKeplerDisk(
            nTotal = 2000,
            vx = -s,
            vy = 0.0,
            x = Config.WIDTH_PX * 0.5,
            y = Config.HEIGHT_PX * 0.6,
            r = 100.0
        )

        return (disc1 + disc2).toMutableList()
    }

    private var engine = PhysicsEngine(defaultBodies())

    private val timer = Timer(1) { tick() } // ~60 FPS (1ms таймер — частая перерисовка)
    private var paused = false
    private var zoom = 1.0 // Initial zoom level

    init {
        preferredSize = Dimension(Config.WIDTH_PX, Config.HEIGHT_PX)
        background = Color.BLACK
        isFocusable = true
        setupKeys()
        setupMouse()
        timer.start()
    }

    // ------ Мышь: ЛКМ добавляет новый диск в точке клика ------
    private fun setupMouse() {
        val mouse = object : MouseAdapter() {
            override fun mousePressed(e: MouseEvent) {
                if (e.button == MouseEvent.BUTTON1) {
                    dragStart = e.point
                    dragCurrent = e.point
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
                if (e.button == MouseEvent.BUTTON1 && dragStart != null) {
                    val start = dragStart!!
                    val end = e.point ?: start
                    val dx = (end.x - start.x).toDouble()
                    val dy = (end.y - start.y).toDouble()
                    val vx = dx * VEL_PER_PIXEL / zoom
                    val vy = dy * VEL_PER_PIXEL / zoom

                    addKeplerDiskAt(start.x.toDouble() / zoom, start.y.toDouble() / zoom, Config.r, Config.n, vx, vy)

                    dragStart = null
                    dragCurrent = null
                    repaint()
                }
            }
            override fun mouseWheelMoved(e: MouseWheelEvent) {
                if (e.getWheelRotation() < 0) {
                    zoom *= 1.1; // Zoom in
                } else {
                    zoom *= 0.9; // Zoom out
                }
                repaint(); // Repaint the panel
            }

        }
        addMouseListener(mouse)
        addMouseMotionListener(mouse) // ← важно: иначе mouseDragged не придёт
        addMouseWheelListener(mouse);
    }


    private fun addKeplerDiskAt(x: Double, y: Double, r: Double, n: Int, vx: Double = initialVY, vy: Double = initialVX) {
        // создаём новый диск в точке клика
        val newDisk = BodyFactory.makeKeplerDisk(
            nTotal = n,
            vx = vx,
            vy = vy,
            x = x,
            y = y,
            r = r
        )
        // объединяем с текущими телами
        val merged = (engine.getBodies() + newDisk).toMutableList()
        engine.resetBodies(merged)
    }

    // ------ Клавиши ------
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

        bind("A") { Config.n = (Config.n - 100).coerceAtLeast(1000) }
        bind("S") { Config.n = (Config.n + 100).coerceAtMost(10000) }

        bind("Q") { Config.r = (Config.r - 10.0).coerceAtLeast(100.0) }
        bind("W") { Config.r = (Config.r + 10.0).coerceAtMost(500.0) }

        bind("O") { Config.DT = (Config.DT - 0.001).coerceAtLeast(-0.015) }
        bind("P") { Config.DT = (Config.DT + 0.001).coerceAtMost(0.015) }

        bind("K") { Config.G = (Config.G - 1.0).coerceAtLeast(0.0) }
        bind("L") { Config.G = (Config.G + 1.0).coerceAtMost(100.0) }

        // Полный перезапуск текущей сцены (один диск по центру)
        bind("R") {  engine.resetBodies(defaultBodies()) }
        bind("ESCAPE") { exitProcess(0) }
    }

    private fun resetSingleCenterDisk(n: Int) {
        val newBodies = BodyFactory.makeKeplerDisk(
            nTotal = n,
            vx = 0.0,
            vy = 0.0,
            x = Config.WIDTH_PX * 0.5,
            y = Config.HEIGHT_PX * 0.5,
            r = clickDiskRadius
        )
        engine.resetBodies(newBodies)
    }

    // ------ Тик симуляции ------
    private fun tick() {
        if (!paused) engine.step()
        repaint()
    }

    // ------ Рендер ------
    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        val g2 = g as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)

        // HUD
        g2.color = Color(0, 255, 0)
        g2.drawString("SPACE — pause | R — reset space | MOUSE1 DRAG'N'DROP — add kepler disk | ESCAPE — exit", 10, 20)
        g2.drawString("Disk radius [Q/W] = ${Config.r}", 10, 60)
        g2.drawString("Bodies count [A/S] = ${Config.n}", 10, 80)
        g2.drawString("Theta [Z/X] = ${Config.theta}", 10, 100)
        g2.drawString("Delta time [O/P] = ${Config.DT}", 10, 120)
        g2.drawString("Gravity [K/L] = ${Config.G}", 10, 140)
        g2.drawString("Softening = ${Config.SOFTENING}", 10, 160)
        g2.drawString("Bodies count = ${engine.getBodies().size}", 10, 180)

        g2.scale(zoom, zoom)

        // Точки тел
        for (b in engine.getBodies()) {
            g2.color = if (b.m >= Config.CENTRAL_MASS) Color.BLACK else Color.WHITE
            val ix = b.x.toInt(); val iy = b.y.toInt()
            if (ix in 0 until (width / zoom).toInt() && iy in 0 until (height / zoom).toInt()) g2.drawLine(ix, iy, ix, iy)
        }

        // Линия от места нажатия до текущей позиции (drag preview)
        if (dragStart != null && dragCurrent != null) {
            val sx = (dragStart!!.x / zoom).toInt(); val sy = (dragStart!!.y / zoom).toInt()
            val ex = (dragCurrent!!.x / zoom).toInt(); val ey = (dragCurrent!!.y / zoom).toInt()
            val oldStroke = g2.stroke
            g2.color = Color(0, 255, 0, 200)
            g2.stroke = BasicStroke(
                1.5f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                10f, floatArrayOf(6f, 6f), 0f
            )
            g2.drawLine(sx, sy, ex, ey)
            g2.stroke = oldStroke
        }

        Toolkit.getDefaultToolkit().sync()
    }
}
