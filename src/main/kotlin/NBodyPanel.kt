import java.awt.Color
import java.awt.Dimension
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.Point
import java.awt.RenderingHints
import java.awt.Toolkit
import java.awt.event.MouseAdapter
import java.awt.event.MouseEvent
import javax.swing.AbstractAction
import javax.swing.JPanel
import javax.swing.KeyStroke
import javax.swing.Timer

// =======================================================
//                    В И З У А Л И З А Ц И Я
//            (отображение, горячие клавиши, HUD)
//     ЛКМ — создать новый Kepler-диск в точке клика
// =======================================================
class NBodyPanel : JPanel() {

    // drag state
    private var dragStart: Point? = null
    private var dragCurrent: Point? = null
    private val showPreview = true
    private val VEL_PER_PIXEL = 1   // 1 px перетаскивания = 0.05 ед. скорости (подбери под себя)
    private val MAX_SPAWN_SPEED = 100.0

    // Базовые настройки для создаваемых кликом дисков
    private val clickDiskRadius = 100.0
    private val initialVX = 0.0
    private val initialVY = 0.0

    // Стартовая сцена: один статичный диск по центру окна
    private var engine = PhysicsEngine(mutableListOf())

    private val timer = Timer(1) { tick() } // ~60 FPS (1ms таймер — частая перерисовка)
    private var paused = false

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
        addMouseListener(object : MouseAdapter() {
            override fun mousePressed(e: MouseEvent) {
                if (e.button == MouseEvent.BUTTON1) {
                    dragStart = e.point
                    dragCurrent = e.point
//                    addKeplerDiskAt(e.x.toDouble(), e.y.toDouble(), 100.0, 2000)
                }
                if (e.button == MouseEvent.BUTTON2) {
                    addKeplerDiskAt(e.x.toDouble(), e.y.toDouble(), 300.0, 4000)
                }
                if (e.button == MouseEvent.BUTTON3) {
                    engine.resetBodies(mutableListOf())
                }
            }
            override fun mouseDragged(e: MouseEvent) {
                if (dragStart != null) {
                    dragCurrent = e.point
                }
            }
            override fun mouseReleased(e: MouseEvent) {
                if (e.button == MouseEvent.BUTTON1 && dragStart != null) {
                    val start = dragStart!!
                    val end = e.point ?: start
                    val dx = (end.x - start.x).toDouble()
                    val dy = (end.y - start.y).toDouble()
                    val vx = dx * VEL_PER_PIXEL
                    val vy = dy * VEL_PER_PIXEL
                    addKeplerDiskAt(start.x.toDouble(), start.y.toDouble(), 100.0, 2000, vx, vy)
                    dragStart = null
                    dragCurrent = null
                }
            }
        })
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

        // Изменение количества тел для будущих создаваемых дисков/перезапуска
        // ВНИМАНИЕ: предполагается, что Config.BODIES_COUNT — var
        bind("A")  {
            Config.BODIES_COUNT = (Config.BODIES_COUNT - 500).coerceAtLeast(50)
            resetSingleCenterDisk(Config.BODIES_COUNT)
        }
        bind("S") {
            Config.BODIES_COUNT = (Config.BODIES_COUNT + 500).coerceAtMost(20000)
            resetSingleCenterDisk(Config.BODIES_COUNT)
        }

        // Полный перезапуск текущей сцены (один диск по центру)
        bind("R") { resetSingleCenterDisk(engine.getBodies().size) }
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
        g2.color = Color.WHITE
        for (b in engine.getBodies()) {
            val ix = b.x.toInt()
            val iy = b.y.toInt()
            if (ix in 0 until width && iy in 0 until height) {
                g2.drawLine(ix, iy, ix, iy) // 1 пиксель
            }
        }
        g2.color = Color(255, 255, 255, 255)
        g2.drawString(
            "N=${engine.getBodies().size}  θ=%.2f  dt=%.3f  G=%.1f  [%s]".format(
                Config.theta, Config.DT, Config.G, if (paused) "PAUSE" else "RUN"
            ),
            10, 20
        )
        g2.drawString("SPACE=pause | Z/X θ± | A/S N± | R=reset | ЛКМ — добавить диск", 10, 36)
        Toolkit.getDefaultToolkit().sync()
    }
}
