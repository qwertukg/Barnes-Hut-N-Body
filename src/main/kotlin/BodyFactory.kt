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

    fun makeGalaxyDisk(
        nTotal: Int,
        // --- ключ к самозакрутке ---
        epsM2: Double = 0.03,              // амплитуда крошечного m=2-возмущения (2–4% достаточно)
        phi0: Double = 0.0,                // фаза «бара» (смещение по углу)
        barTaperR: Double? = null,         // радиус затухания возмущения; по умолчанию берем rMax*0.6
        // профиль/«холодность» диска
        radialScale: Double? = null,       // масштаб экспоненты Rd; по умолчанию rMax/3
        speedJitter: Double = 0.01,        // маленький шум скоростей, чтобы диск был «холодным»
        radialJitter: Double = 0.0,        // лучше 0 — радиальных скоростей почти нет
        clockwise: Boolean = true,
        rng: Random = Random(Random.nextLong()),
        vx: Double = 0.0, vy: Double = 0.0,
        x: Double = Config.WIDTH_PX * 0.5,
        y: Double = Config.HEIGHT_PX * 0.5,
        r: Double = 200.0,
        minR: Double = Config.MIN_R,
        centralMass: Double = Config.CENTRAL_MASS,
        totalSatelliteMass: Double = Config.TOTAL_SATELLITE_MASS
    ): MutableList<Body> {
        val cx = x
        val cy = y
        val rMax = r
        val sats = (nTotal - 1).coerceAtLeast(0)
        val bodies = ArrayList<Body>(sats + 1)

        // центр (для самозакрутки обычно полезно уменьшить CENTRAL_MASS и поднять массу диска в Config)
        bodies += Body(cx, cy, vx, vy, centralMass)

        val mSat = if (sats > 0) totalSatelliteMass / sats else 0.0
        val Rd = radialScale ?: (rMax / 3.0)
        val taperR = barTaperR ?: (rMax * 0.6)

        // выбор R по экспоненциальному профилю на [minR, rMax]
        fun sampleExpRadius(): Double {
            val u = rng.nextDouble()
            val A = exp(-(rMax - minR) / Rd)
            val t = 1 - u * (1 - A)
            return minR - Rd * ln(t)
        }

        // расстановка звёзд: почти осесимметрично + микроскопическое m=2 возмущение по радиусу
        repeat(sats) {
            val R = sampleExpRadius()
            val theta = rng.nextDouble() * 2.0 * Math.PI

            // m=2 «яйцеобразность» (бар): r' = r*(1 + eps*cos(2*(θ-φ0))*taper(R))
            val taper = exp(- (R / taperR) * (R / taperR)) // мягкое затухание возмущения к периферии
            val R2 = R * (1.0 + epsM2 * cos(2.0 * (theta - phi0)) * taper)

            val px = cx + R2 * cos(theta)
            val py = cy + R2 * sin(theta)
            bodies += Body(px, py, 0.0, 0.0, mSat)
        }

        // точный M_enclosed по фактическим позициям (как у тебя)
        data class RIdx(val i: Int, val r: Double)
        val sorted = bodies.mapIndexed { i, b -> RIdx(i, hypot(b.x - cx, b.y - cy)) }.sortedBy { it.r }
        var acc = 0.0
        val Menc = DoubleArray(bodies.size)
        for (ri in sorted) { acc += bodies[ri.i].m; Menc[ri.i] = acc }

        // круговые скорости + минимальный шум; радиальных скоростей почти нет (холодный диск)
        for (i in 1 until bodies.size) {
            val b = bodies[i]
            val dx = b.x - cx; val dy = b.y - cy
            val R = max(1e-6, hypot(dx, dy))
            val vCirc = sqrt(Config.G * Menc[i] / R)
            val v = vCirc * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * speedJitter)

            // чисто тангенциальная компонента (холодно); при желании можно добавить крошечный vr ~ radialJitter*vCirc
            val (tx, ty) = if (clockwise) (dy / R) to (-dx / R) else (-dy / R) to (dx / R)
            var vx0 = tx * v
            var vy0 = ty * v

            if (radialJitter > 0.0) {
                // очень небольшая радиальная скорость вдоль (dx,dy)
                val vr = (rng.nextDouble() - 0.5) * 2.0 * radialJitter * vCirc
                vx0 += (dx / R) * vr
                vy0 += (dy / R) * vr
            }

            b.vx = vx0 + vx
            b.vy = vy0 + vy
        }

        return bodies
    }

    /**
     * Равномерно и независимо размещает по всей области экрана [0..WIDTH_PX)×[0..HEIGHT_PX)
     * n одинаковых тел с массой m. Начальные скорости — нулевые.
     *
     * @param n   количество тел
     * @param m   масса каждого тела (должна быть > 0)
     * @param rng генератор случайных чисел (по умолчанию фиксированный seed для повторяемости)
     */
    fun makeUniformRandom(
        n: Int,
        m: Double,
        rng: Random = Random(Random.nextLong())
    ): MutableList<Body> {
        if (n <= 0 || m <= 0.0) return mutableListOf()

        val bodies = ArrayList<Body>(n)
        val w = Config.WIDTH_PX.toDouble()
        val h = Config.HEIGHT_PX.toDouble()

        repeat(n) {
            val x = rng.nextDouble() * w
            val y = rng.nextDouble() * h
            bodies += Body(x, y, 0.0, 0.0, m)
        }
        return bodies
    }

}