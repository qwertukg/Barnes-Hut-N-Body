import java.awt.Color
import java.awt.Dimension
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.RenderingHints
import java.awt.Toolkit
import javax.swing.AbstractAction
import javax.swing.JPanel
import javax.swing.KeyStroke
import javax.swing.Timer

// =======================================================
//                    В И З У А Л И З А Ц И Я
//            (отображение, горячие клавиши, HUD)
// =======================================================
class NBodyPanel : JPanel() {
    val initialVX = 0.0
    val initialVY = 0.0
    private var engine = PhysicsEngine(
        when (Config.scene) {
            Config.Scene.KEPLER_DISK   -> {
                BodyFactory.makeKeplerDisk(
                    Config.BODIES_COUNT,
                    vx = 0.0,
                    x = Config.WIDTH_PX * 0.5,
                    r = 300.0
                )
            }
            Config.Scene.KEPLER_DISK_COLLIDER -> {
                val disk1 = BodyFactory.makeKeplerDisk(
                    Config.BODIES_COUNT,
                    vx = initialVX,
                    x = Config.WIDTH_PX * 0.3,
                    r = 300.0
                )
                val disk2 = BodyFactory.makeKeplerDisk(
                    Config.BODIES_COUNT/2,
                    vx = initialVX,
                    x = Config.WIDTH_PX * 0.7,
                    y = Config.HEIGHT_PX * 0.4,
                    r = 200.0
                )
                (disk1 + disk2).toMutableList()
            }
        }
    )

    private val timer = Timer(1) { tick() } // ~60 FPS
    private var paused = false

    init {
        preferredSize = Dimension(Config.WIDTH_PX, Config.HEIGHT_PX)
        background = Color.BLACK
        isFocusable = true
        setupKeys()
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
        bind("Z") { Config.theta = (Config.theta - 0.05).coerceAtLeast(0.2) }
        bind("X") { Config.theta = (Config.theta + 0.05).coerceAtMost(1.6) }
        bind("A")  {
            Config.BODIES_COUNT = (Config.BODIES_COUNT - 500).coerceAtLeast(50)
            resetWithSameScene(Config.BODIES_COUNT)
        }
        bind("S") {
            Config.BODIES_COUNT = (Config.BODIES_COUNT + 500).coerceAtMost(20000)
            resetWithSameScene(Config.BODIES_COUNT)
        }
        bind("R") { resetWithSameScene(engine.getBodies().size) }
        // Переключение сцен
        bind("1") { Config.scene = Config.Scene.KEPLER_DISK;  resetWithSameScene(engine.getBodies().size) }
        bind("2") { Config.scene = Config.Scene.KEPLER_DISK_COLLIDER; resetWithSameScene(engine.getBodies().size) }
    }

    private fun resetWithSameScene(n: Int) {
        val newBodies = when (Config.scene) {
            Config.Scene.KEPLER_DISK   -> {
                BodyFactory.makeKeplerDisk(
                    n,
                    vx = 10.0,
                    x = Config.WIDTH_PX * 0.5,
                    r = 300.0
                )
            }
            Config.Scene.KEPLER_DISK_COLLIDER -> {
                val disk1 = BodyFactory.makeKeplerDisk(
                    n,
                    vx = initialVX,
                    x = Config.WIDTH_PX * 0.3,
                    r = 300.0
                )
                val disk2 = BodyFactory.makeKeplerDisk(
                    n/2,
                    vx = initialVX,
                    x = Config.WIDTH_PX * 0.7,
                    y = Config.HEIGHT_PX * 0.4,
                    r = 200.0
                )
                (disk1 + disk2).toMutableList()
            }
        }
        engine.resetBodies(newBodies)
    }

    private fun tick() {
        if (!paused) engine.step()
        repaint()
    }

    override fun paintComponent(g: Graphics) {
        super.paintComponent(g)
        val g2 = g as Graphics2D
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_OFF)
        g2.color = Color.WHITE
        for (b in engine.getBodies()) {
            val ix = b.x.toInt(); val iy = b.y.toInt()
            if (ix in 0 until width && iy in 0 until height) g2.drawLine(ix, iy, ix, iy) // 1 px
        }
        g2.color = Color(255, 255, 255, 255)
        g2.drawString(
            "N=${engine.getBodies().size}  θ=%.2f  dt=%.3f  G=%.1f  scene=%s  [%s]".format(
                Config.theta, Config.DT, Config.G, Config.scene, if (paused) "PAUSE" else "RUN"
            ),
            10, 20
        )
        g2.drawString("SPACE=pause | Z/X θ± | A/S N± | R=reset | 1=Kepler static  2=Kepler collide".trim(), 10, 36)
        Toolkit.getDefaultToolkit().sync()
    }
}