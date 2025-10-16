package barneshut

import kotlin.math.max
import kotlin.math.sqrt

/** Узел квадродерева Barnes–Hut. */
internal class BHTree(private val quad: Quad) {
    private var leafBody: Body? = null
    private var children: Array<BHTree?>? = null

    var mass: Double = 0.0
        private set
    var centerX: Double = quad.cx
        private set
    var centerY: Double = quad.cy
        private set

    private fun isLeaf(): Boolean = children == null

    fun insert(body: Body) {
        if (!quad.contains(body)) return
        if (leafBody == null && isLeaf()) {
            leafBody = body
            return
        }
        if (isLeaf()) subdivide()
        leafBody?.let { existing ->
            leafBody = null
            insertIntoChild(existing)
        }
        insertIntoChild(body)
    }

    private fun insertIntoChild(body: Body) {
        if (quad.halfSize < 1e-3) {
            val epsilon = 1e-3
            val bitX = body.x.toBits() and 1L
            val bitY = body.y.toBits() and 1L
            body.x += if (bitX == 0L) epsilon else -epsilon
            body.y += if (bitY == 0L) -epsilon else epsilon
        }
        val array = children ?: return
        val ix = if (body.x < quad.cx) 0 else 1
        val iy = if (body.y < quad.cy) 0 else 2
        val childIndex = ix + iy
        array[childIndex]?.insert(body)
    }

    private fun subdivide() {
        children = arrayOf(
            BHTree(quad.child(0)),
            BHTree(quad.child(1)),
            BHTree(quad.child(2)),
            BHTree(quad.child(3)),
        )
    }

    fun computeMass() {
        if (isLeaf()) {
            val body = leafBody
            if (body != null) {
                mass = body.m
                centerX = body.x
                centerY = body.y
            } else {
                mass = 0.0
                centerX = quad.cx
                centerY = quad.cy
            }
            return
        }
        var massSum = 0.0
        var cx = 0.0
        var cy = 0.0
        val array = children ?: return
        for (child in array) {
            child?.computeMass()
            if (child != null && child.mass > 0.0) {
                massSum += child.mass
                cx += child.centerX * child.mass
                cy += child.centerY * child.mass
            }
        }
        mass = massSum
        if (massSum > 0.0) {
            centerX = cx / massSum
            centerY = cy / massSum
        } else {
            centerX = quad.cx
            centerY = quad.cy
        }
    }

    fun accumulateForce(target: Body, config: SimulationConfig, acc: ForceAcc) {
        if (mass == 0.0) return
        val dx = centerX - target.x
        val dy = centerY - target.y
        val distSq = dx * dx + dy * dy + config.softeningSquared
        val distance = sqrt(distSq)
        if (isLeaf()) {
            val body = leafBody ?: return
            if (body === target) return
            addForceContribution(dx, dy, distance, acc, config)
            return
        }
        val size = quad.halfSize * 2.0
        if (size / max(distance, 1e-6) < config.theta) {
            addForceContribution(dx, dy, distance, acc, config)
        } else {
            val array = children ?: return
            for (child in array) {
                child?.accumulateForce(target, config, acc)
            }
        }
    }

    private fun addForceContribution(
        dx: Double,
        dy: Double,
        distance: Double,
        acc: ForceAcc,
        config: SimulationConfig
    ) {
        val invDist = if (distance > 0.0) 1.0 / distance else 0.0
        val invDist3 = invDist * invDist * invDist
        val force = config.g * mass * invDist3
        acc.fx += force * dx
        acc.fy += force * dy
    }
}
