package barneshut

/** Состояние тела в плоскости. */
data class Body(
    var x: Double,
    var y: Double,
    var vx: Double,
    var vy: Double,
    var m: Double
)

/** Вспомогательный аккумулятор силы. */
internal class ForceAcc {
    var fx: Double = 0.0
    var fy: Double = 0.0

    fun reset() {
        fx = 0.0
        fy = 0.0
    }
}
