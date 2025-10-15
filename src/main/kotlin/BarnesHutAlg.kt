import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Описание состояния частицы гравитационной системы Barnes–Hut.
 * @property x текущая координата по оси X (пиксели экрана).
 * @property y текущая координата по оси Y (пиксели экрана).
 * @property vx скорость вдоль оси X (пиксели в секунду).
 * @property vy скорость вдоль оси Y (пиксели в секунду).
 * @property m масса частицы в условных единицах.
 */
data class Body(
    var x: Double, var y: Double,
    var vx: Double, var vy: Double,
    var m: Double
)

/**
 * Минимальный аккумулятор компонент силы, переиспользуемый воркерами.
 * Значения очищаются на каждый просчёт конкретного тела.
 */
class Acc {
    /** Накопленная сила по X. */
    var fx = 0.0

    /** Накопленная сила по Y. */
    var fy = 0.0

    /** Сбросить накопитель перед новым расчётом сил. */
    fun reset() { fx = 0.0; fy = 0.0 }
}

/**
 * Квадрат Barnes–Hut с центром и половиной стороны.
 * Область служит для рекурсивного деления пространства частиц.
 */
data class Quad(val cx: Double, val cy: Double, val h: Double) {
    /** Проверить попадание тела в область квадрата. */
    fun contains(b: Body): Boolean =
        b.x >= cx - h && b.x < cx + h && b.y >= cy - h && b.y < cy + h

    /** Вернуть дочерний квадрат по индексу (0..3). */
    fun child(which: Int): Quad {
        val hh = h / 2.0 // половина текущей стороны, задаёт размер дочерней области
        return when (which) {
            0 -> Quad(cx - hh, cy - hh, hh) // NW — левый верхний квадрант
            1 -> Quad(cx + hh, cy - hh, hh) // NE — правый верхний квадрант
            2 -> Quad(cx - hh, cy + hh, hh) // SW — левый нижний квадрант
            else -> Quad(cx + hh, cy + hh, hh) // SE — правый нижний квадрант
        }
    }
}

/**
 * Узел квадродерева Barnes–Hut, агрегирующий массу и центр масс дочерних узлов.
 * @property quad описываемый пространственный квадрат.
 */
class BHTree(private val quad: Quad) {
    private var body: Body? = null
    private var children: Array<BHTree?>? = null
    var mass: Double = 0.0; private set
    var comX: Double = 0.0; private set
    var comY: Double = 0.0; private set

    /** Проверка, что узел является листом (без детей). */
    private fun isLeaf(): Boolean = children == null

    /**
     * Вставить новое тело в дерево, расширяя ветку по необходимости.
     * @param b добавляемое тело.
     */
    fun insert(b: Body) {
        if (!quad.contains(b)) return // пропускаем тело, если оно находится вне границ квадрата
        if (body == null && isLeaf()) {
            body = b // заполняем лист, если он пустой
            return
        }
        if (isLeaf()) subdivide() // лист с телом превращаем в внутренний узел
        body?.let { existing ->
            body = null // переносим существующее тело вниз по дереву
            insertIntoChild(existing)
        }
        insertIntoChild(b) // рекурсивно вставляем новое тело
    }

    /**
     * Вставить тело в подходящий дочерний квадрант, компенсируя числовые проблемы.
     * @param b добавляемое тело.
     */
    private fun insertIntoChild(b: Body) {
        // защита от деградации при совпадающих координатах — без Random в горячем пути
        if (quad.h < 1e-3) {
            val eps = 1e-3 // минимальное смещение для разведения точек
            b.x += if ((b.x.toBits() and 1L) == 0L) +eps else -eps
            b.y += if ((b.y.toBits() and 1L) == 0L) -eps else +eps
        }
        val ch = children!! // дочерние узлы гарантированно созданы к этому моменту
        val ix = if (b.x < quad.cx) 0 else 1 // горизонтальный индекс квадранта
        val iy = if (b.y < quad.cy) 0 else 2 // вертикальный индекс квадранта
        ch[ix + iy]!!.insert(b) // рекурсивно добавляем тело в нужный сектор
    }

    /** Разбить узел на четыре дочерних квадранта. */
    private fun subdivide() {
        children = arrayOf(
            BHTree(quad.child(0)),
            BHTree(quad.child(1)),
            BHTree(quad.child(2)),
            BHTree(quad.child(3)),
        )
    }

    /**
     * Пересчитать суммарную массу и центр масс поддерева.
     * Листовые узлы берут значения непосредственно из тела либо центра квадрата.
     */
    fun computeMass() {
        if (isLeaf()) {
            body?.let {
                mass = it.m; comX = it.x; comY = it.y // центр масс совпадает с координатами тела
            } ?: run { mass = 0.0; comX = quad.cx; comY = quad.cy } // пустой лист — масса 0 в центре квадрата
        } else {
            var mSum = 0.0; var cx = 0.0; var cy = 0.0 // аккумуляторы массы и момента
            val ch = children!!
            ch[0]!!.computeMass(); if (ch[0]!!.mass > 0.0) { mSum += ch[0]!!.mass; cx += ch[0]!!.comX * ch[0]!!.mass; cy += ch[0]!!.comY * ch[0]!!.mass }
            ch[1]!!.computeMass(); if (ch[1]!!.mass > 0.0) { mSum += ch[1]!!.mass; cx += ch[1]!!.comX * ch[1]!!.mass; cy += ch[1]!!.comY * ch[1]!!.mass }
            ch[2]!!.computeMass(); if (ch[2]!!.mass > 0.0) { mSum += ch[2]!!.mass; cx += ch[2]!!.comX * ch[2]!!.mass; cy += ch[2]!!.comY * ch[2]!!.mass }
            ch[3]!!.computeMass(); if (ch[3]!!.mass > 0.0) { mSum += ch[3]!!.mass; cx += ch[3]!!.comX * ch[3]!!.mass; cy += ch[3]!!.comY * ch[3]!!.mass }
            mass = mSum
            if (mSum > 0.0) { comX = cx / mSum; comY = cy / mSum } else { comX = quad.cx; comY = quad.cy }
        }
    }

    /**
     * Быстрый расчёт точечной силы без лишних аллокаций.
     * @param b тело, на которое действует сила.
     * @param px координата X источника силы.
     * @param py координата Y источника силы.
     * @param m масса источника.
     * @param acc аккумулятор сил.
     */
    private fun pointForceAcc(b: Body, px: Double, py: Double, m: Double, acc: Acc) {
        val dx = px - b.x // смещение по X от тела до источника силы
        val dy = py - b.y // смещение по Y от тела до источника силы
        val r2 = dx*dx + dy*dy + Config.SOFT2 // квадрат расстояния с учётом софтенинга
        val invR = 1.0 / sqrt(r2) // обратная величина расстояния
        val invR2 = 1.0 / r2 // обратная величина квадрата расстояния
        val f = Config.G * b.m * m * invR2 // модуль силы по закону Ньютона
        acc.fx += f * dx * invR // проекция силы на X
        acc.fy += f * dy * invR // проекция силы на Y
    }

    /** Накопить силу на b в acc. Критерий Барнса–Хатта в квадратах: s^2 < θ^2 * dist^2 */
    fun accumulateForce(b: Body, theta2: Double, acc: Acc) {
        if (mass == 0.0) return // пустые узлы не влияют на силу
        if (isLeaf()) {
            val single = body
            if (single == null || single === b) return // не взаимодействуем с самим собой
            pointForceAcc(b, comX, comY, mass, acc)
            return
        }
        val dx = comX - b.x // смещение от тела до центра масс поддерева по X
        val dy = comY - b.y // смещение от тела до центра масс поддерева по Y
        val dist2 = dx*dx + dy*dy + Config.SOFT2 // расстояние до поддерева с софтенингом
        val s2 = (quad.h * 2.0).let { it * it } // квадрат длины стороны квадранта

        if (s2 < theta2 * dist2) {
            pointForceAcc(b, comX, comY, mass, acc) // поддерево достаточно далеко, аппроксимируем точечной массой
        } else {
            val ch = children!! // иначе спускаемся глубже
            ch[0]!!.accumulateForce(b, theta2, acc)
            ch[1]!!.accumulateForce(b, theta2, acc)
            ch[2]!!.accumulateForce(b, theta2, acc)
            ch[3]!!.accumulateForce(b, theta2, acc)
        }
    }

    /** Обойти все квадраты дерева (корень → потомки). */
    fun visitQuads(visit: (Quad) -> Unit) {
        visit(quad)
        val ch = children
        if (ch != null) {
            ch[0]?.visitQuads(visit)
            ch[1]?.visitQuads(visit)
            ch[2]?.visitQuads(visit)
            ch[3]?.visitQuads(visit)
        }
    }

}

/**
 * Физический движок, управляющий шагами симуляции и параллельным расчётом сил.
 * @param initialBodies стартовый набор тел.
 */
class PhysicsEngine(initialBodies: MutableList<Body>) {
    private val cores = Runtime.getRuntime().availableProcessors()
    private var bodies: MutableList<Body> = initialBodies
    private var ax = DoubleArray(bodies.size)
    private var ay = DoubleArray(bodies.size)

    /** Последнее построенное дерево (для отрисовки/отладки). */
    private var lastTree: BHTree? = null

    fun getTreeForDebug(): BHTree {
        val t = lastTree
        return t ?: buildTree().also { lastTree = it }
    }

    /** Получить текущий неизменяемый список тел. */
    fun getBodies(): List<Body> = bodies

    /** Полностью заменить набор тел, обновив буферы ускорений. */
    fun resetBodies(newBodies: MutableList<Body>) {
        bodies = newBodies
        if (ax.size != bodies.size) { ax = DoubleArray(bodies.size); ay = DoubleArray(bodies.size) }
        lastTree = null // сброс кэша дерева
    }

    /**
     * Построить квадродерево для текущего множества тел.
     * Корневой квадрат охватывает всю область окна с небольшим запасом.
     */
    private fun buildTree(): BHTree {
        val half = max(Config.WIDTH_PX, Config.HEIGHT_PX) / 2.0 + 2.0 // половина стороны охватывающего квадрата
        val root = BHTree(Quad(Config.WIDTH_PX / 2.0, Config.HEIGHT_PX / 2.0, half)) // корневой узел дерева
        val bs = bodies // локальная ссылка на список тел, чтобы избежать повторных обращений к свойству
        for (b in bs) root.insert(b) // добавляем каждое тело в дерево
        root.computeMass() // вычисляем центры масс и суммарные массы
        return root
    }

    /** Параллельный расчёт ускорений (фиксированное число воркеров). */
    private suspend fun computeAccelerations(root: BHTree) = coroutineScope {
        val bs = bodies // ссылка на список тел для ускорения доступа
        val n = bs.size // количество тел в системе
        val workers = min(cores, n.coerceAtLeast(1)) // число параллельных задач
        val theta2 = Config.theta * Config.theta // критерий Барнса–Хатта в квадрате
        val next = AtomicInteger(0) // индексация задач воркерами

        repeat(workers) {
            launch(Dispatchers.Default) {
                val acc = Acc() // переиспользуем в рамках воркера
                while (true) {
                    val i = next.getAndIncrement() // забираем очередной индекс тела
                    if (i >= n) break // выход, если тело закончилось
                    val b = bs[i] // текущее тело
                    acc.reset()
                    root.accumulateForce(b, theta2, acc)
                    ax[i] = acc.fx / b.m // ускорение по X
                    ay[i] = acc.fy / b.m // ускорение по Y
                }
            }
        }
    }

    /** Один шаг Leapfrog (kick–drift–kick). */
    fun step() {
        // слияние тел после шага
        mergeCloseBodiesIfNeeded()

        var root = buildTree() // строим дерево для текущего положения
        runBlocking { computeAccelerations(root) } // вычисляем ускорения на текущий момент

        val bs = bodies // список тел для удобного обращения
        val dtHalf = Config.DT * 0.5 // половина шага по времени
        for (i in bs.indices) {
            bs[i].vx += ax[i] * dtHalf // первый «kick»: корректировка скорости по X
            bs[i].vy += ay[i] * dtHalf // первый «kick»: корректировка скорости по Y
        }

        for (b in bs) {
            b.x += b.vx * Config.DT // «drift»: перенос по X
            b.y += b.vy * Config.DT // «drift»: перенос по Y
        }

        root = buildTree() // дерево для обновлённых позиций
        runBlocking { computeAccelerations(root) } // ускорения на конце шага

        for (i in bs.indices) {
            bs[i].vx += ax[i] * dtHalf // второй «kick»: завершаем обновление скорости по X
            bs[i].vy += ay[i] * dtHalf // второй «kick»: завершаем обновление скорости по Y
        }

        // сохранить дерево для рендера
        lastTree = root


    }

    /**
     * Слить близкие тела с «прожорливыми» массивными телами.
     *
     * Правило:
     *  - Если у тела i масса > mergeMaxMass и оно находится ближе mergeMinDist к телу j,
     *    то масса i увеличивается на массу j, а тело j удаляется.
     *  - Если массы равны (i.m == j.m), удаляется второе тело (j).
     *
     * Замечания:
     *  - Проверяется евклидово расстояние.
     *  - Удаления производятся на лету; индексы аккуратно поддерживаются.
     *  - Буферы ax/ay не пересоздаём — на следующем шаге будут использоваться только
     *    первые bodies.size элементов.
     */
    private fun mergeCloseBodiesIfNeeded() {
        val maxM = mergeMaxMass
        val minD = mergeMinDist
        if (minD <= 0.0) return
        if (bodies.size <= 1) return

        val minD2 = minD * minD
        var i = 0
        while (i < bodies.size) {
            val bi = bodies[i]
            if (bi.m > maxM) {
                var j = i + 1
                while (j < bodies.size) {
                    val bj = bodies[j]
                    val dx = bj.x - bi.x
                    val dy = bj.y - bi.y
                    if (dx*dx + dy*dy < minD2) {
                        // столкновение по правилу
                        bi.m += bj.m
                        bodies.removeAt(j)
                        continue
                    }
                    j++
                }
            }
            i++
        }

        // так как состав тел изменился — сбросим кэш дерева
        lastTree = null
    }

    /**
     * Порог массы «прожорливого» тела. Только тела с массой > mergeMaxMass
     * поглощают соседей на дистанции < mergeMinDist.
     *
     * Значение задайте снаружи перед шагами симуляции.
     */
    var mergeMaxMass: Double = 4_000.0

    /**
     * Порог дистанции для слияния (евклидово расстояние).
     *
     * Значение задайте снаружи перед шагами симуляции.
     */
    var mergeMinDist: Double = Config.MIN_R
}
