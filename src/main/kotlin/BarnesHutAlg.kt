import kotlinx.coroutines.Deferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Состояние частицы в 2D-симуляции гравитации по методу Barnes–Hut.
 *
 * @property x Текущая координата по оси X (в пикселях экрана/мировых единицах визуализации).
 * @property y Текущая координата по оси Y (в пикселях).
 * @property vx Текущая скорость по оси X (пикселей в секунду).
 * @property vy Текущая скорость по оси Y (пикселей в секунду).
 * @property m Масса частицы в условных единицах.
 */
data class Body(
    var x: Double, var y: Double,
    var vx: Double, var vy: Double,
    var m: Double
)

/**
 * Небольшой «аккумулятор» компонент силы, переиспользуемый внутри рабочих задач.
 *
 * Он создан, чтобы избежать лишних аллокаций объектов при суммировании вкладов силы
 * от большого числа узлов квадродерева к одному телу.
 */
class Acc {
    /** Накопленная проекция силы на ось X. */
    var fx = 0.0
    /** Накопленная проекция силы на ось Y. */
    var fy = 0.0

    /** Сбросить значения перед новым расчётом силы для очередного тела. */
    fun reset() { fx = 0.0; fy = 0.0 }
}

/**
 * Квадрат области пространства (квадрант) для квадродерева Barnes–Hut.
 *
 * @property cx Координата центра квадрата по X.
 * @property cy Координата центра квадрата по Y.
 * @property h Полуполовина стороны квадрата (т.е. половина половины полной длины стороны).
 *
 * Квадрант задан центром и «полу-половиной» стороны, поэтому вся область квадрата —
 * это [cx−h, cx+h) × [cy−h, cy+h).
 */
data class Quad(val cx: Double, val cy: Double, val h: Double) {

    /**
     * Проверка попадания тела в квадрат.
     *
     * @return true, если тело находится внутри полуинтервалов по X и Y
     *         (левая/верхняя границы включены, правая/нижняя — исключены).
     */
    fun contains(b: Body): Boolean =
        b.x >= cx - h && b.x < cx + h && b.y >= cy - h && b.y < cy + h

    /**
     * Сформировать дочерний квадрант.
     *
     * @param which Индекс 0..3:
     *  - 0: NW (левый верх)
     *  - 1: NE (правый верх)
     *  - 2: SW (левый низ)
     *  - 3: SE (правый низ)
     */
    fun child(which: Int): Quad {
        val hh = h / 2.0 // размер дочернего квадранта
        return when (which) {
            0 -> Quad(cx - hh, cy - hh, hh) // NW
            1 -> Quad(cx + hh, cy - hh, hh) // NE
            2 -> Quad(cx - hh, cy + hh, hh) // SW
            else -> Quad(cx + hh, cy + hh, hh) // SE
        }
    }
}

/**
 * Узел квадродерева Barnes–Hut.
 *
 * Каждый узел хранит:
 *  - либо одно тело (лист),
 *  - либо ссылки на 4 дочерних узла и агрегированную информацию:
 *    суммарную массу поддерева и координаты его центра масс.
 *
 * @constructor
 * @param quad Квадрант пространства, который покрывает данный узел.
 */
class BHTree(private val quad: Quad) {
    /** Если узел — лист и в нём есть тело, оно хранится здесь. */
    private var body: Body? = null

    /** Массив из 4 дочерних узлов (или null, если узел — лист). */
    private var children: Array<BHTree?>? = null

    /** Суммарная масса тела/поддерева, которое покрывает узел. */
    var mass: Double = 0.0; private set

    /** X-координата центра масс тела/поддерева, которое покрывает узел. */
    var comX: Double = 0.0; private set

    /** Y-координата центра масс тела/поддерева, которое покрывает узел. */
    var comY: Double = 0.0; private set

    /** Признак, что узел — лист (нет дочерних узлов). */
    private fun isLeaf(): Boolean = children == null

    /**
     * Вставить тело в деревo.
     *
     * Алгоритм:
     *  1. Если тело вне квадранта узла — игнорируем.
     *  2. Если узел — пустой лист — кладём тело сюда.
     *  3. Иначе:
     *     - если узел был листом, делим его на 4 дочерних (subdivide);
     *     - перекладываем «старое» тело ниже по иерархии (если было);
     *     - рекурсивно вставляем новое тело в соответствующий дочерний узел.
     */
    fun insert(b: Body) {
        if (!quad.contains(b)) return
        if (body == null && isLeaf()) {
            body = b
            return
        }
        if (isLeaf()) subdivide()
        body?.let { existing ->
            body = null
            insertIntoChild(existing)
        }
        insertIntoChild(b)
    }

    /**
     * Вставить тело в подходящий ребёнок, с защитой от деградации при совпадающих координатах.
     *
     * Когда размер квадранта становится очень маленьким (h < 1e-3),
     * слегка раздвигаем совпадающие точки на эпсилон, без аллокации Random.
     */
    private fun insertIntoChild(b: Body) {
        if (quad.h < 1e-3) {
            val eps = 1e-3
            // Небольшие детерминированные сдвиги на основе младшего бита мантиссы:
            b.x += if ((b.x.toBits() and 1L) == 0L) +eps else -eps
            b.y += if ((b.y.toBits() and 1L) == 0L) -eps else +eps
        }
        val ch = children!!
        val ix = if (b.x < quad.cx) 0 else 1
        val iy = if (b.y < quad.cy) 0 else 2
        ch[ix + iy]!!.insert(b)
    }

    /** Превратить лист в внутренний узел: создать 4 дочерних квадранта. */
    private fun subdivide() {
        children = arrayOf(
            BHTree(quad.child(0)),
            BHTree(quad.child(1)),
            BHTree(quad.child(2)),
            BHTree(quad.child(3)),
        )
    }

    /**
     * Подсчитать агрегаты массы:
     *  - для листа: масса/центр масс = массы/координаты тела (или 0/центр квадранта, если тела нет);
     *  - для узла с детьми: сумма масс детей, центр масс как средневзвешенный.
     */
    fun computeMass() {
        if (isLeaf()) {
            body?.let {
                mass = it.m
                comX = it.x
                comY = it.y
            } ?: run {
                mass = 0.0
                comX = quad.cx
                comY = quad.cy
            }
        } else {
            var mSum = 0.0
            var cx = 0.0
            var cy = 0.0
            val ch = children!!
            ch[0]!!.computeMass(); if (ch[0]!!.mass > 0.0) { mSum += ch[0]!!.mass; cx += ch[0]!!.comX * ch[0]!!.mass; cy += ch[0]!!.comY * ch[0]!!.mass }
            ch[1]!!.computeMass(); if (ch[1]!!.mass > 0.0) { mSum += ch[1]!!.mass; cx += ch[1]!!.comX * ch[1]!!.mass; cy += ch[1]!!.comY * ch[1]!!.mass }
            ch[2]!!.computeMass(); if (ch[2]!!.mass > 0.0) { mSum += ch[2]!!.mass; cx += ch[2]!!.comX * ch[2]!!.mass; cy += ch[2]!!.comY * ch[2]!!.mass }
            ch[3]!!.computeMass(); if (ch[3]!!.mass > 0.0) { mSum += ch[3]!!.mass; cx += ch[3]!!.comX * ch[3]!!.mass; cy += ch[3]!!.comY * ch[3]!!.mass }
            mass = mSum
            if (mSum > 0.0) {
                comX = cx / mSum
                comY = cy / mSum
            } else {
                comX = quad.cx
                comY = quad.cy
            }
        }
    }

    /**
     * Добавить вклад силы на тело [b] от этого узла.
     *
     * Ветвление по критерию Барнса–Хатта:
     *  - если узел достаточно «мал и далёк», считаем его точечной массой (один вызов [pointForceAcc]);
     *  - иначе спускаемся к детям и суммируем их вклад рекурсивно.
     *
     * @param b Тело, для которого накапливаем силу.
     * @param theta2 Порог тэта^2 (используем квадрат, чтобы не вычислять корни).
     * @param acc Аккумулятор силы (переиспользуется снаружи).
     */
    fun accumulateForce(b: Body, theta2: Double, acc: Acc) {
        if (mass == 0.0) return
        if (isLeaf()) {
            val single = body
            if (single == null || single === b) return // пропускаем пустой лист и «само-взаимодействие»
            pointForceAcc(b, comX, comY, mass, acc)
            return
        }
        val dx = comX - b.x
        val dy = comY - b.y
        val dist2 = dx*dx + dy*dy + Config.SOFT2 // софтенинг в критерии
        val s2 = (quad.h * 2.0).let { it * it } // квадрат длины стороны квадранта

        if (s2 < theta2 * dist2) {
            // далеко → аппроксимируем одну точку
            pointForceAcc(b, comX, comY, mass, acc)
        } else {
            // близко/крупно → раскрываемся на 4 ребёнка
            val ch = children!!
            ch[0]!!.accumulateForce(b, theta2, acc)
            ch[1]!!.accumulateForce(b, theta2, acc)
            ch[2]!!.accumulateForce(b, theta2, acc)
            ch[3]!!.accumulateForce(b, theta2, acc)
        }
    }

    /**
     * Быстрый расчёт гравитационной силы точки массы [m], расположенной в ([px], [py]),
     * действующей на тело [b]. Результат прибавляется в аккумулятор [acc].
     *
     * Используются:
     *  - один sqrt (для 1/r),
     *  - одна обратная величина r² (для 1/r²),
     *  - минимально возможное число делений.
     */
    private fun pointForceAcc(b: Body, px: Double, py: Double, m: Double, acc: Acc) {
        val dx = px - b.x
        val dy = py - b.y
        val r2 = dx*dx + dy*dy + Config.SOFT2         // r² с софтенингом
        val invR = 1.0 / sqrt(r2)                      // 1/r
        val invR2 = 1.0 / r2                           // 1/r²
        val f = Config.G * b.m * m * invR2             // модуль силы
        acc.fx += f * dx * invR                        // fx = f * cos = f * (dx/r)
        acc.fy += f * dy * invR                        // fy = f * sin = f * (dy/r)
    }

    /**
     * Обойти дерево (корень → потомки) и вызвать [visit] для каждого квадранта.
     * Используется для отладки/визуализации границ квадродерева.
     */
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
 * Физический «движок» симуляции:
 *  - строит квадродерево,
 *  - считает силы параллельно,
 *  - интегрирует уравнения движения (Leapfrog: kick–drift–kick),
 *  - выполняет опциональные слияния тел (rule-based merge).
 *
 * @constructor
 * @param initialBodies Стартовый набор тел.
 */
class PhysicsEngine(initialBodies: MutableList<Body>) {

    // ------------------ Конфигурация и рабочие буферы ------------------

    /** Количество аппаратных логических ядер (для подбора числа воркеров). */
    private val cores = Runtime.getRuntime().availableProcessors()

    /** Текущий изменяемый список тел. */
    private var bodies: MutableList<Body> = initialBodies

    /** Рабочий буфер ускорений по X (длина поддерживается не меньше числа тел). */
    private var ax = DoubleArray(bodies.size)

    /** Рабочий буфер ускорений по Y (длина поддерживается не меньше числа тел). */
    private var ay = DoubleArray(bodies.size)

    /** Последнее построенное квадродерево (кэш для отрисовки/отладки). */
    private var lastTree: BHTree? = null

    // ---------------------- Параметры «пожирания» ----------------------

    /**
     * Порог массы «прожорливого» тела.
     * Только тела с массой **строго больше** этого значения поглощают соседей
     * на дистанции меньше [mergeMinDist].
     *
     * Значение должно устанавливаться вызывающим кодом (например, UI) перед шагами.
     */
    var mergeMaxMass: Double = 4_000.0

    /**
     * Пороговая дистанция (в пикселях) для слияния тел.
     * Если 0 или отрицательное — слияние отключено.
     */
    var mergeMinDist: Double = Config.MIN_R

    // ------------------------- Публичные API ----------------------------

    /**
     * Вернуть квадродерево для отладки/визуализации.
     * Если кэш пуст — будет построено свежее дерево.
     */
    fun getTreeForDebug(): BHTree {
        val t = lastTree
        return t ?: buildTree().also { lastTree = it }
    }

    /** Текущие тела системы (read-only интерфейс). */
    fun getBodies(): List<Body> = bodies

    /**
     * Полностью заменить набор тел.
     * Буферы ускорений переразмеряются при необходимости (без усечения).
     * Кэш дерева сбрасывается.
     */
    fun resetBodies(newBodies: MutableList<Body>) {
        bodies = newBodies
        if (ax.size != bodies.size) {
            ax = DoubleArray(bodies.size)
            ay = DoubleArray(bodies.size)
        }
        lastTree = null
    }

    // ----------------------- Внутренняя механика -----------------------

    /**
     * Построить квадродерево для текущего набора тел.
     *
     * Корневой квадрат размещается по центру окна ([Config.WIDTH_PX]/[Config.HEIGHT_PX])
     * и выбирается достаточно большим, чтобы покрыть всю сцену.
     */
    private fun buildTree(): BHTree {
        val half = max(Config.WIDTH_PX, Config.HEIGHT_PX) / 2.0 + 2.0
        val root = BHTree(Quad(Config.WIDTH_PX / 2.0, Config.HEIGHT_PX / 2.0, half))
        val bs = bodies
        for (b in bs) root.insert(b)
        root.computeMass()
        return root
    }

    /**
     * Параллельно вычислить ускорения всех тел на основе квадродерева [root].
     *
     * Приёмные буферы [ax]/[ay] заполняются синхронно с индексом тела.
     * Количество воркеров ограничено числом ядер и числом тел.
     */
    private suspend fun computeAccelerations(root: BHTree) = coroutineScope {
        val bs = bodies
        val n = bs.size
        val workers = min(cores, n.coerceAtLeast(1))
        val theta2 = Config.theta * Config.theta // используем θ², чтобы не делать sqrt
        val next = AtomicInteger(0)              // «раздача» индексов задачам

        repeat(workers) {
            launch(Dispatchers.Default) {
                val acc = Acc()
                while (true) {
                    val i = next.getAndIncrement()
                    if (i >= n) break
                    val b = bs[i]
                    acc.reset()
                    root.accumulateForce(b, theta2, acc)
                    ax[i] = acc.fx / b.m
                    ay[i] = acc.fy / b.m
                }
            }
        }
    }

    /**
     * Один интеграционный шаг схемой Leapfrog (kick–drift–kick):
     *
     * 1) Считаем a(t) и делаем «полукик» скорости v(t+dt/2).
     * 2) Сдвигаем координаты («drift») x(t+dt) по v(t+dt/2).
     * 3) Считаем a(t+dt) и завершаем скорость до v(t+dt).
     * 4) Обновляем кэш дерева и запускаем правило слияния (если активировано).
     */
    fun step() {
        // a(t)
        var root = buildTree()
        runBlocking { computeAccelerations(root) }

        // kick: v(t+dt/2)
        val bs = bodies
        val dtHalf = Config.DT * 0.5
        for (i in bs.indices) {
            bs[i].vx += ax[i] * dtHalf
            bs[i].vy += ay[i] * dtHalf
        }

        // drift: x(t+dt)
        for (b in bs) {
            b.x += b.vx * Config.DT
            b.y += b.vy * Config.DT
        }

        // a(t+dt)
        root = buildTree()
        runBlocking { computeAccelerations(root) }

        // kick: v(t+dt)
        for (i in bs.indices) {
            bs[i].vx += ax[i] * dtHalf
            bs[i].vy += ay[i] * dtHalf
        }

        // сохранить дерево для рендера/отладки
        lastTree = root

        // выполнить правило «пожирания» (если включено настройками)
        mergeCloseBodiesIfNeeded()
    }

    // -------------------------- MERGE-ПРАВИЛО --------------------------

    /**
     * Слить близкие тела с «прожорливыми» массивными телами.
     *
     * Правило:
     *  - Для каждого тела **i** с массой `i.m > mergeMaxMass` ищутся *все* тела **j ≠ i**,
     *    такие что расстояние `|i−j| < mergeMinDist`.
     *  - Все найденные j удаляются, их масса добавляется к `i.m`.
     *  - Если массы i и j совпадают, удаляется **второе** тело (j) — как оговорено в постановке.
     *
     * Особенности реализации:
     *  - Поиск «жертв» распараллелен по диапазону индексов (корутины на `Dispatchers.Default`).
     *  - Индексы жертв собираются в списки, конкатенируются и сортируются по убыванию
     *    перед фактическим удалением (чтобы не сбивать индексацию).
     *  - После удаления переоцениваем позицию **i** с помощью `indexOf(bi)`, так как
     *    при удалениях элементов слева от i его индекс мог уменьшиться.
     *  - По завершении сбрасывается кэш квадродерева ([lastTree] = null).
     *
     * Отключение:
     *  - Чтобы полностью отключить слияния, установите `mergeMinDist <= 0`.
     */
    private fun mergeCloseBodiesIfNeeded() {
        // ранний выход: отключено или слишком мало тел
        if (mergeMinDist <= 0.0 || bodies.size <= 1) return

        // Рабочая величина: квадрат пороговой дистанции (без sqrt в цикле)
        val minD2 = mergeMinDist * mergeMinDist

        var i = 0
        while (i < bodies.size) {
            val bi = bodies[i] // кандидат на «прожорливость»

            if (bi.m > mergeMaxMass) {
                val n = bodies.size
                if (n > 1) {
                    // Подготовка распараллеливания по всему диапазону [0, n) (исключая j == i)
                    val workers = min(cores, max(1, n / 4096))
                    val step = max(1, (n + workers - 1) / workers)

                    // Собираем индексы жертв (victims) параллельно чанками
                    val victims: List<Int> = runBlocking {
                        val jobs = ArrayList<Deferred<MutableList<Int>>>(workers)
                        var s = 0
                        while (s < n) {
                            val e = min(n, s + step)
                            val s0 = s
                            val e0 = e
                            jobs += async(Dispatchers.Default) {
                                val local = ArrayList<Int>((e0 - s0).coerceAtLeast(0))
                                var j = s0
                                while (j < e0) {
                                    if (j != i) {                // не сравниваем тело с самим собой
                                        val bj = bodies[j]
                                        val dx = bj.x - bi.x
                                        val dy = bj.y - bi.y
                                        if (dx * dx + dy * dy < minD2) {
                                            // Жертва найдена: по условию «если равны — удаляем второе»
                                            // логика удаления одинакова: «поглощаем j»
                                            local.add(j)
                                        }
                                    }
                                    j++
                                }
                                local
                            }
                            s = e
                        }
                        buildList { for (d in jobs) addAll(d.await()) }
                    }

                    // Применяем поглощение: удаляем жертв, начиная с наибольших индексов
                    if (victims.isNotEmpty()) {
                        for (j in victims.sortedDescending()) {
                            if (j < 0 || j >= bodies.size) continue // на случай гонок индексации
                            if (bodies[j] === bi) continue          // защита от самоуказания
                            val bj = bodies[j]
                            bi.m += bj.m
                            bodies.removeAt(j)
                        }
                        // Индекс bi мог сдвинуться из-за удалений слева → нормализуем i
                        val newIndex = bodies.indexOf(bi)
                        i = if (newIndex >= 0) newIndex else (i - 1).coerceAtLeast(0)

                        // Структура изменилась — сбрасываем кэш дерева
                        lastTree = null
                    }
                }
            }
            i++
        }
    }
}