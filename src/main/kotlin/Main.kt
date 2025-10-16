import java.awt.Toolkit
import javax.swing.*

/** Точка входа приложения визуализации Barnes–Hut. */
fun main() {
    val useFullscreen = Config.FULL_SCREEN_MODE
    SwingUtilities.invokeLater {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName())

        val screen = Toolkit.getDefaultToolkit().screenSize // актуальные размеры экрана
        Config.WIDTH_PX = screen.width // настраиваем ширину окна под экран
        Config.HEIGHT_PX = screen.height // настраиваем высоту окна под экран

        val panel = NBodyPanel() // основная панель с визуализацией

        val frame = JFrame("Barnes–Hut N-Body") // контейнер для панели
        frame.isUndecorated = useFullscreen // отключаем рамки окна для иммерсивности
        frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
        frame.contentPane.add(panel)

        val ge = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment() // доступ к графической среде
        val gd = ge.defaultScreenDevice // основной экран пользователя

        try {
            if (useFullscreen && gd.isFullScreenSupported) {
                gd.fullScreenWindow = frame // эксклюзивный fullscreen
            } else {
                frame.extendedState = JFrame.MAXIMIZED_BOTH // fallback на максимизацию
                frame.isVisible = true
            }
        } catch (_: Exception) {
            // fallback
            frame.extendedState = JFrame.MAXIMIZED_BOTH
            frame.isVisible = true
        }
    }
}
