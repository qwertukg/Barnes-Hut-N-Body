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
import javax.swing.AbstractAction
import javax.swing.JPanel
import javax.swing.KeyStroke
import javax.swing.Timer
import kotlin.system.exitProcess

/** Панель Swing, отвечающая за визуализацию и управление симуляцией Barnes–Hut. */
class NBodyPanel : JPanel() {

    /** Точка начала перетаскивания мышью. */
    private var dragStart: Point? = null

    /** Текущее положение курсора во время перетаскивания. */
    private var dragCurrent: Point? = null

    /** Перевод пикселей перетаскивания в скорость. */
    private val VEL_PER_PIXEL = 1   // 1 px перетаскивания = 0.05 ед. скорости (подбери под себя)

    /** Базовая скорость по X для кликовых дисков. */
    private val initialVX = 0.0

    /** Базовая скорость по Y для кликовых дисков. */
    private val initialVY = 0.0

    /** Показывать ли границы квадродерева. */
    private var showTree = false

    /**
     * Сформировать стартовый набор тел: два кеплеровских диска с противоположным дрейфом.
     * Следуем рекомендациям DAML по симметричной постановке задачи.
     */
    fun defaultBodies(): MutableList<Body> {
        val s = 70.0  // s — модуль «дрейфовой» скорости диска (px/сек), задаёт движение всего диска целиком

        // Первый диск:
        val disc1 = BodyFactory.makeKeplerDisk(
            nTotal = Config.N,                 // nTotal — сколько тел создать в этом диске (включая центральное)
            vx = s,                        // vx — добавочный сдвиг скорости по X для всех тел диска (px/сек)
            vy = 0.0,                      // vy — добавочный сдвиг скорости по Y для всех тел диска (px/сек)
            x = Config.WIDTH_PX * 0.5,     // x — координата центра диска по X (в пикселях экрана)
            y = Config.HEIGHT_PX * 0.36,    // y — координата центра диска по Y (в пикселях экрана)
            r = Config.R,                     // r — радиус диска (макс. расстояние частиц от центра) в пикселях
            clockwise = true              // clockwise — вращение по часовой
        )

        // Второй диск:
        val disc2 = BodyFactory.makeKeplerDisk(
            nTotal = Config.N,                 // число тел во втором диске
            vx = -s,                       // двигать диск влево (противоположное направление первому)
            vy = 0.0,                      // вертикального дрейфа нет
            x = Config.WIDTH_PX * 0.5,     // центр по X тот же
            y = Config.HEIGHT_PX * 0.64,    // центр по Y ниже, чтобы диски шли навстречу
            r = Config.R,                      // радиус второго диска
            clockwise = true
        )

        // Склеиваем оба списка тел в один
        return (disc1 + disc2).toMutableList()
    }


    /** Движок симуляции для текущего множества тел. */
    private var engine = PhysicsEngine(defaultBodies())

    /** Периодический таймер перерисовки и шагов симуляции. */
    private val timer = Timer(1) { tick() } // ~60 FPS (1ms таймер — частая перерисовка)

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

    /** Настроить обработку мыши для добавления новых дисков. */
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
                    val start = dragStart!! // сохранённая точка начала перетаскивания
                    val end = e.point ?: start // конечная точка (или начало, если недоступно)
                    val dx = (end.x - start.x).toDouble() // смещение по X
                    val dy = (end.y - start.y).toDouble() // смещение по Y
                    val vx = dx * VEL_PER_PIXEL // результирующая скорость по X
                    val vy = dy * VEL_PER_PIXEL // результирующая скорость по Y

                    addKeplerDiskAt(start.x.toDouble(), start.y.toDouble(), Config.R, Config.N, vx, vy)

                    dragStart = null
                    dragCurrent = null
                    repaint()
                }
            }
        }
        addMouseListener(mouse)
        addMouseMotionListener(mouse) // ← важно: иначе mouseDragged не придёт
    }


    /**
     * Добавить новый кеплеровский диск по указанным параметрам.
     * @param x координата центра по X.
     * @param y координата центра по Y.
     * @param r радиус диска.
     * @param n число тел.
     * @param vx добавочная скорость по X.
     * @param vy добавочная скорость по Y.
     */
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

    /** Настроить горячие клавиши для управления симуляцией. */
    private fun setupKeys() {
        fun bind(key: String, action: () -> Unit) {
            val im = getInputMap(WHEN_IN_FOCUSED_WINDOW) // карта привязок клавиш
            val am = actionMap // карта действий панели
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

        bind("O") { Config.DT = (Config.DT - 0.001).coerceAtLeast(-0.015) }
        bind("P") { Config.DT = (Config.DT + 0.001).coerceAtMost(0.015) }

        bind("K") { Config.G = (Config.G - 1.0).coerceAtLeast(0.0) }
        bind("L") { Config.G = (Config.G + 1.0).coerceAtMost(100.0) }

        // Полный перезапуск текущей сцены (один диск по центру)
        bind("R") {  engine.resetBodies(defaultBodies()) }
        bind("ESCAPE") { exitProcess(0) }

        // показать/скрыть границы квадродерева
        bind("D") { showTree = !showTree; repaint() }
    }

    /** Один кадр визуализации и, при необходимости, шаг симуляции. */
    private fun tick() {
        if (!paused) engine.step()
        repaint()
    }

    /** Отрисовать все тела и служебные элементы HUD. */
    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        val g2 = g as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)

        // Точки тел
        for (b in engine.getBodies()) {
            g2.color = if (b.m >= Config.CENTRAL_MASS) Color.BLACK else Color.WHITE
            val ix = b.x.toInt(); val iy = b.y.toInt()
            if (ix in 0 until width && iy in 0 until height) g2.drawLine(ix, iy, ix, iy)
        }

        // Линия от места нажатия до текущей позиции (drag preview)
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
            val w = (Config.R * 2).toInt()
            g2.drawArc(sx - Config.R.toInt(), sy - Config.R.toInt(), w, w, 0, 360)
            g2.stroke = oldStroke
        }

        // границы квадродерева
        if (showTree) {
            val oldColor = g2.color
            val oldStroke = g2.stroke
            g2.color = Color(0, 255, 0, 180)
            g2.stroke = BasicStroke(1f)

            // Получаем последнее дерево (если пауза/старт — построим разок)
            val tree = engine.getTreeForDebug()
            tree.visitQuads { q ->
                val x = (q.cx - q.h).toInt()
                val y = (q.cy - q.h).toInt()
                val s = (q.h * 2).toInt()
                g2.drawLine(x, y, x, y+s)
                g2.drawLine(x, y, x+s, y)
            }

            g2.color = oldColor
            g2.stroke = oldStroke
        }

        // HUD
        g2.color = Color(0, 255, 0)
        g2.drawString("SPACE — pause | R — reset space | MOUSE1 DRAG'N'DROP — add kepler disk | ESCAPE — exit", 10, 20)
        g2.drawString("Disk radius [Q/W] = ${Config.R}", 10, 60)
        g2.drawString("Bodies count [A/S] = ${Config.N}", 10, 80)
        g2.drawString("Theta [Z/X] = ${Config.theta}", 10, 100)
        g2.drawString("Delta time [O/P] = ${Config.DT}", 10, 120)
        g2.drawString("Gravity [K/L] = ${Config.G}", 10, 140)
        g2.drawString("Debug mode = $showTree", 10, 160)
        g2.drawString("Bodies count = ${engine.getBodies().size}", 10, 180)
        g2.drawString("Softening = ${Config.SOFTENING}", 10, 200)

        Toolkit.getDefaultToolkit().sync()
    }
}
