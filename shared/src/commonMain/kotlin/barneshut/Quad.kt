package barneshut

/** Квадрант плоскости, описываемый центром и половиной длины стороны. */
internal data class Quad(
    val cx: Double,
    val cy: Double,
    val halfSize: Double
) {
    fun contains(body: Body): Boolean =
        body.x >= cx - halfSize && body.x < cx + halfSize &&
            body.y >= cy - halfSize && body.y < cy + halfSize

    fun child(index: Int): Quad {
        val quarter = halfSize / 2.0
        return when (index) {
            0 -> Quad(cx - quarter, cy - quarter, quarter) // NW
            1 -> Quad(cx + quarter, cy - quarter, quarter) // NE
            2 -> Quad(cx - quarter, cy + quarter, quarter) // SW
            else -> Quad(cx + quarter, cy + quarter, quarter) // SE
        }
    }
}
