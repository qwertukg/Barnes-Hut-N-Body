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
    private var engine = PhysicsEngine(
        when (Config.scene) {
            Config.Scene.KEPLER_DISK   -> BodyFactory.makeKeplerDisk(Config.BODIES_COUNT)
            Config.Scene.SELF_GRAV_DISK -> BodyFactory.makeSelfGravDisk(Config.BODIES_COUNT)
        }
    )
    private val timer = Timer(16) { tick() } // ~60 FPS
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
        bind("OPEN_BRACKET")  { resetWithSameScene((engine.getBodies().size * 0.9).toInt().coerceAtLeast(50)) }
        bind("CLOSE_BRACKET") { resetWithSameScene((engine.getBodies().size * 1.1).toInt().coerceAtMost(20000)) }
        bind("R") { resetWithSameScene(engine.getBodies().size) }
        // Переключение сцен
        bind("1") { Config.scene = Config.Scene.KEPLER_DISK;  resetWithSameScene(engine.getBodies().size) }
        bind("2") { Config.scene = Config.Scene.SELF_GRAV_DISK; resetWithSameScene(engine.getBodies().size) }
    }

    private fun resetWithSameScene(n: Int) {
        val newBodies = when (Config.scene) {
            Config.Scene.KEPLER_DISK    -> BodyFactory.makeKeplerDisk(n)
            Config.Scene.SELF_GRAV_DISK -> BodyFactory.makeSelfGravDisk(n)
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
        g2.color = Color(255, 255, 255, 200)
        g2.drawString(
            "N=${engine.getBodies().size}  θ=%.2f  dt=%.3f  G=%.1f  scene=%s  [%s]".format(
                Config.theta, Config.DT, Config.G, Config.scene, if (paused) "PAUSE" else "RUN"
            ),
            10, 20
        )
        g2.drawString("SPACE=pause | Z/X θ± | [ ] N± | R=reset | 1=Kepler  2=Self-grav".trim(), 10, 36)
        Toolkit.getDefaultToolkit().sync()
    }
}