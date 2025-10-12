import kotlin.math.*
import kotlin.random.Random

/** Фабрика генерации наборов тел для различных стартовых сценариев. */
object BodyFactory {

    /**
     * Кеплеровский диск: центр + спутники с v_circ по M_enclosed(r).
     * В соответствии с каноном DAML формирует массово-скоростное распределение с учётом enclosed mass.
     */
    fun makeKeplerDisk(
        nTotal: Int,
        clockwise: Boolean = true,
        radialJitter: Double = 0.03,
        speedJitter: Double = 0.01,
        rng: Random = Random(3),
        vx: Double = 0.0,
        vy: Double = 0.0,
        x: Double = Config.WIDTH_PX * 0.5,
        y: Double = Config.HEIGHT_PX * 0.5,
        r: Double = min(Config.WIDTH_PX, Config.HEIGHT_PX) * 0.38
    ): MutableList<Body> {
        val cx = x // координата центра диска по X
        val cy = y // координата центра диска по Y
        val rMax = r // максимальный радиус диска
        val sats = (nTotal - 1).coerceAtLeast(0) // число спутников без центрального тела
        val bodies = ArrayList<Body>(sats + 1) // итоговый список тел

        bodies += Body(cx, cy, vx, vy, Config.CENTRAL_MASS) // центральное массивное тело

        val mSat = if (sats > 0) Config.TOTAL_SATELLITE_MASS / sats else 0.0 // масса каждого спутника

        repeat(sats) {
            val u = rng.nextDouble() // равномерная величина для радиального распределения
            val rr = sqrt(u * (rMax * rMax - Config.MIN_R * Config.MIN_R) + Config.MIN_R * Config.MIN_R) // радиус до джиттера
            val rJ = rr * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * radialJitter) // радиус с джиттером
            val ang = rng.nextDouble() * 2.0 * Math.PI // угол в полярных координатах
            val x = cx + rJ * cos(ang) // позиция спутника по X
            val y = cy + rJ * sin(ang) // позиция спутника по Y
            bodies += Body(x, y, 0.0, 0.0, mSat)
        }

        data class RIdx(val i: Int, val r: Double) // индекс тела и расстояние до центра
        val sorted = bodies.mapIndexed { i, b -> RIdx(i, hypot(b.x - cx, b.y - cy)) }.sortedBy { it.r } // сортировка по радиусу
        var acc = 0.0 // аккумулятор массы внутри радиуса
        val Menc = DoubleArray(bodies.size) // массив enclosed mass
        for (ri in sorted) { acc += bodies[ri.i].m; Menc[ri.i] = acc }

        for (i in 1 until bodies.size) {
            val b = bodies[i]
            val dx = b.x - cx; val dy = b.y - cy // вектор от центра к спутнику
            val r = max(1e-6, hypot(dx, dy)) // расстояние до центра
            val vCirc = sqrt(Config.G * Menc[i] / r) // идеальная круговая скорость
            val v = vCirc * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * speedJitter) // скорость с джиттером
            val (tx, ty) = if (clockwise) (dy / r) to (-dx / r) else (-dy / r) to (dx / r) // орт нормали направления
            b.vx = tx * v; b.vy = ty * v // присваиваем тангенциальную скорость
            b.vx += vx // добавляем дрейфовую скорость по X
            b.vy += vy // добавляем дрейфовую скорость по Y
        }
        return bodies
    }
}