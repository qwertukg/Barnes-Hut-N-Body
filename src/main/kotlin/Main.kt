import java.awt.Toolkit
import javax.swing.*

fun main() {
    javax.swing.SwingUtilities.invokeLater {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName())

        val screen = Toolkit.getDefaultToolkit().screenSize
        Config.WIDTH_PX = screen.width
        Config.HEIGHT_PX = screen.height

        val panel = NBodyPanel()

        val frame = JFrame("Barnes–Hut N-Body • Kotlin + Coroutines")
        frame.isUndecorated = true
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        frame.contentPane.add(panel)

        val ge = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment()
        val gd = ge.defaultScreenDevice

        try {
            if (gd.isFullScreenSupported) {
                gd.fullScreenWindow = frame // эксклюзивный fullscreen
            } else {
                frame.extendedState = JFrame.MAXIMIZED_BOTH
                frame.isVisible = true
            }
        } catch (_: Exception) {
            // fallback
            frame.extendedState = JFrame.MAXIMIZED_BOTH
            frame.isVisible = true
        }
    }
}
