import javax.swing.*

fun main() {
    SwingUtilities.invokeLater {
        UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName())
        JFrame("Barnes–Hut N-Body • Kotlin + Coroutines").apply {
            defaultCloseOperation = JFrame.EXIT_ON_CLOSE
            contentPane.add(NBodyPanel())
            pack()
            setLocationRelativeTo(null)
            isVisible = true
        }
    }
}
