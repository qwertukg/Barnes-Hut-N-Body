@file:JvmName("GpuNBodySSBORender")
package gpu

import org.lwjgl.glfw.GLFW.*
import org.lwjgl.glfw.GLFWErrorCallback
import org.lwjgl.opengl.GL
import org.lwjgl.opengl.GL46.*
import org.lwjgl.system.MemoryUtil.*
import kotlin.math.*
import kotlin.random.Random

/**
 * Global configuration for windowing, compute, physics, rendering, and galaxy generation defaults.
 */
private object Config {

    // --- Window / GL context ---

    /** Window width in pixels for the GLFW context. */
    const val WIDTH = 3440

    /** Window height in pixels for the GLFW context. */
    const val HEIGHT = 1440


    // --- Compute / workgroup ---

    /** Local workgroup size for the compute shader (x-dimension). */
    const val WORK_GROUP_SIZE = 256


    // --- Physics parameters ---

    /** Gravitational constant scale used by the compute shader. */
    const val G = 80.0f

    /** Fixed physics timestep (seconds). */
    const val DT = 0.005f

    /** Plummer-like softening length (added as soft^2 to distance^2). */
    const val SOFTENING = 1.0f


    // --- Rendering parameters ---

    /** Base GL point size (in pixels) before mass scaling. */
    const val POINT_SIZE = 1f

    /** Additional size contribution per unit of mass (0 disables mass-based sizing). */
    const val MASS_POINT_SCALE = 0.0f

    /** Dark background flag (true for dark gray, false for white). */
    const val BACKGROUND_DARK = true


    // --- Convenience (double-precision mirrors) ---

    /** Width in pixels as Double (for helpers that prefer doubles). */
    const val WIDTH_PX = WIDTH.toDouble()

    /** Height in pixels as Double (for helpers that prefer doubles). */
    const val HEIGHT_PX = HEIGHT.toDouble()


    // --- Galaxy generation defaults (makeGalaxyDisk) ---

    /** Minimum orbital radius clamp to avoid singularities (pixels). */
    const val MIN_R = 2.0

    /** Mass of the central body used by disk galaxy generator. */
    const val CENTRAL_MASS = 5_000.0

    /** Total mass distributed across satellite bodies in the disk galaxy. */
    const val TOTAL_SATELLITE_MASS = 25_000.0
}


/**
 * Immutable body definition used for CPU-side generation and upload.
 *
 * @property x X position.
 * @property y Y position.
 * @property z Z position.
 * @property vx X velocity.
 * @property vy Y velocity.
 * @property vz Z velocity.
 * @property m Mass.
 */
data class Body(
    var x: Float, var y: Float, var z: Float,
    var vx: Float, var vy: Float, var vz: Float,
    var m: Float
)

/**
 * Builds and links the compute shader program used for N-body simulation.
 *
 * @return OpenGL program handle for the compute shader.
 * @throws IllegalStateException if compilation or linking fails.
 */
private fun buildComputeProgram(): Int {
    val src = """
#version 460 core
layout(local_size_x = ${Config.WORK_GROUP_SIZE}) in;

struct Body { vec4 posMass; vec4 velPad; };

layout(std430, binding = 0) buffer BodyBuffer { Body bodies[]; };

uniform float uDt;
uniform float uSoftening;
uniform float uG;
uniform uint  uCount;

shared vec4 tilePosMass[${Config.WORK_GROUP_SIZE}];

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uCount) return;

    uint localIndex = gl_LocalInvocationID.x;
    Body self = bodies[id];
    vec3 position = self.posMass.xyz;
    float mass = self.posMass.w;
    vec3 velocity = self.velPad.xyz;
    vec3 acc = vec3(0.0);

    for (uint tile = 0u; tile < uCount; tile += ${Config.WORK_GROUP_SIZE}u) {
        uint idx = tile + localIndex;
        tilePosMass[int(localIndex)] = (idx < uCount) ? bodies[idx].posMass : vec4(0.0);
        barrier();

        uint tileSize = min(uCount - tile, uint(${Config.WORK_GROUP_SIZE}));
        for (uint j = 0u; j < tileSize; ++j) {
            uint otherIndex = tile + j;
            if (otherIndex == id) continue;
            vec4 other = tilePosMass[int(j)];
            vec3 d = other.xyz - position;
            float dist2 = dot(d,d) + uSoftening;
            float invR = inversesqrt(dist2);
            float invR3 = invR * invR * invR;
            acc += (uG * other.w) * d * invR3;
        }
        barrier();
    }

    velocity += acc * uDt;
    position += velocity * uDt;

    bodies[id].posMass = vec4(position, mass);
    bodies[id].velPad  = vec4(velocity, 0.0);
}
""".trimIndent()

    val shader = glCreateShader(GL_COMPUTE_SHADER)
    glShaderSource(shader, src)
    glCompileShader(shader)
    if (glGetShaderi(shader, GL_COMPILE_STATUS) != GL_TRUE) {
        val log = glGetShaderInfoLog(shader)
        glDeleteShader(shader)
        error("Compute shader compile error:\n$log")
    }
    val prog = glCreateProgram()
    glAttachShader(prog, shader)
    glLinkProgram(prog)
    if (glGetProgrami(prog, GL_LINK_STATUS) != GL_TRUE) {
        val log = glGetProgramInfoLog(prog)
        glDeleteProgram(prog)
        glDeleteShader(shader)
        error("Compute program link error:\n$log")
    }
    glDetachShader(prog, shader)
    glDeleteShader(shader)
    return prog
}

/**
 * Builds and links the render shader program (vertex + fragment) for point rendering.
 *
 * @return OpenGL program handle for the render pipeline.
 * @throws IllegalStateException if compilation or linking fails.
 */
private fun buildRenderProgram(): Int {
    val vs = """
#version 460 core
struct Body { vec4 posMass; vec4 velPad; };
layout(std430, binding = 0) buffer BodyBuffer { Body bodies[]; };

uniform vec2  uViewport;
uniform float uPointBase;
uniform float uMassScale;
uniform float uCamAngle;
uniform float uCamPitch;
uniform vec3  uCenter;
uniform float uSpeedScale;

out float vSpeed;
out float vMass;

void main(){
    int id = gl_VertexID;
    vec3 p = bodies[id].posMass.xyz;
    vec3 v = bodies[id].velPad.xyz;
    float m = bodies[id].posMass.w;

    vec3 q = p - uCenter;

    float ca = cos(uCamAngle);
    float sa = sin(uCamAngle);
    vec3 rY;
    rY.x =  ca * q.x + sa * q.z;
    rY.y =  q.y;
    rY.z = -sa * q.x + ca * q.z;

    float cp = cos(uCamPitch);
    float sp = sin(uCamPitch);
    vec3 pr;
    pr.x = rY.x;
    pr.y =  cp * rY.y - sp * rY.z;
    pr.z =  sp * rY.y + cp * rY.z;

    float x =  pr.x / (uViewport.x * 0.5);
    float y = -pr.y / (uViewport.y * 0.5);
    gl_Position = vec4(x, y, 0.0, 1.0);

    gl_PointSize = max(1.0, uPointBase + uMassScale * m);

    vSpeed = length(v);
    vMass  = m;
}
""".trimIndent()

    val fs = """
#version 460 core
in float vSpeed;
in float vMass;
out vec4 fragColor;

uniform float uSpeedScale;

void main(){
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    if (dot(c,c) > 1.0) discard;

    float t = clamp(vSpeed * uSpeedScale, 0.0, 1.0) * 5.0;

    const float W = 0.77;
    vec3 white = vec3(1.0);
    vec3 slow = mix(white, vec3(1.0, 1.0, 1.0), 1.0 - W);
    vec3 mid  = mix(white, vec3(0.0, 1.0, 1.0), 1.0 - W);
    vec3 fast = mix(white, vec3(0.65, 0.00, 0.95), 1.0 - W);

    vec3 color = mix( mix(slow, mid, smoothstep(0.0, 0.5, t)),
                      fast,         smoothstep(0.5, 1.0, t) );

    fragColor = vec4(color, 1.0);
}
""".trimIndent()

    fun compile(type: Int, src: String): Int {
        val s = glCreateShader(type)
        glShaderSource(s, src)
        glCompileShader(s)
        if (glGetShaderi(s, GL_COMPILE_STATUS) != GL_TRUE) {
            val log = glGetShaderInfoLog(s)
            glDeleteShader(s)
            error("Shader compile error:\n$log")
        }
        return s
    }

    val vsId = compile(GL_VERTEX_SHADER, vs)
    val fsId = compile(GL_FRAGMENT_SHADER, fs)
    val prog = glCreateProgram()
    glAttachShader(prog, vsId)
    glAttachShader(prog, fsId)
    glLinkProgram(prog)
    if (glGetProgrami(prog, GL_LINK_STATUS) != GL_TRUE) {
        val log = glGetProgramInfoLog(prog)
        glDeleteProgram(prog)
        glDeleteShader(vsId)
        glDeleteShader(fsId)
        error("Render program link error:\n$log")
    }
    glDetachShader(prog, vsId)
    glDetachShader(prog, fsId)
    glDeleteShader(vsId)
    glDeleteShader(fsId)
    return prog
}

/**
 * GPU N-body engine wrapper that manages SSBO storage, simulation, and rendering.
 *
 * @property count Number of bodies.
 * @property bodies Initial body list to upload.
 */
private class GpuNBodyRenderer(
    private var count: Int,
    bodies: List<Body>
) : AutoCloseable {

    private val ssbo = glGenBuffers()
    private val vao = glGenVertexArrays()

    private val computeProg = buildComputeProgram()
    private val renderProg = buildRenderProgram()

    private val uDt      = glGetUniformLocation(computeProg, "uDt")
    private val uSoft    = glGetUniformLocation(computeProg, "uSoftening")
    private val uG       = glGetUniformLocation(computeProg, "uG")
    private val uCountC  = glGetUniformLocation(computeProg, "uCount")

    private val uViewport  = glGetUniformLocation(renderProg, "uViewport")
    private val uPointBase = glGetUniformLocation(renderProg, "uPointBase")
    private val uMassScale = glGetUniformLocation(renderProg, "uMassScale")
    private val uCamAngle  = glGetUniformLocation(renderProg, "uCamAngle")
    private val uCenter    = glGetUniformLocation(renderProg, "uCenter")
    private val uCamPitch = glGetUniformLocation(renderProg, "uCamPitch")
    private val uSpeedScale = glGetUniformLocation(renderProg, "uSpeedScale")

    private var capacityBytes = 0L

    /**
     * Initializes GPU buffers and uploads initial bodies.
     */
    init {
        glBindVertexArray(vao)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, 0L, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        ensureCapacity(count)
        uploadBodies(bodies)
    }

    /**
     * Replaces the current body set with a new one and resizes GPU buffers if needed.
     *
     * @param newBodies New body list.
     */
    fun resizeBodies(newBodies: List<Body>) {
        count = newBodies.size
        ensureCapacity(count)
        uploadBodies(newBodies)
    }

    /**
     * Ensures SSBO capacity for at least [n] bodies.
     *
     * @param n Body count to accommodate.
     */
    private fun ensureCapacity(n: Int) {
        val floatsPerBody = 8
        val bytes = n.toLong() * floatsPerBody * java.lang.Float.BYTES
        if (bytes > capacityBytes) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            capacityBytes = bytes
        }
    }

    /**
     * Uploads a body array to the SSBO.
     *
     * @param bodies Body list to upload.
     */
    private fun uploadBodies(bodies: List<Body>) {
        val n = bodies.size
        if (n == 0) return
        val floatsPerBody = 8
        val buf = memAllocFloat(n * floatsPerBody)
        try {
            for (b in bodies) {
                buf.put(b.x).put(b.y).put(b.z).put(b.m)
                buf.put(b.vx).put(b.vy).put(b.vz).put(0f)
            }
            buf.flip()
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, buf)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        } finally { memFree(buf) }
    }

    /**
     * Computes center of mass by reading back SSBO contents.
     *
     * @return Center of mass as [x, y, z].
     */
    fun computeCenterOfMass(): FloatArray {
        if (count == 0) return floatArrayOf(0f, 0f, 0f)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        val floatsPerBody = 8
        val totalFloats = count * floatsPerBody
        val buf = memAllocFloat(totalFloats)
        try {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0L, buf)
            var sx = 0.0; var sy = 0.0; var sz = 0.0; var sm = 0.0
            for (i in 0 until count) {
                val base = i * floatsPerBody
                val x = buf.get(base + 0).toDouble()
                val y = buf.get(base + 1).toDouble()
                val z = buf.get(base + 2).toDouble()
                val m = buf.get(base + 3).toDouble()
                sx += x * m; sy += y * m; sz += z * m; sm += m
            }
            if (sm == 0.0) return floatArrayOf(0f, 0f, 0f)
            return floatArrayOf((sx/sm).toFloat(), (sy/sm).toFloat(), (sz/sm).toFloat())
        } finally { memFree(buf) }
    }

    /**
     * Executes one simulation step on the GPU.
     *
     * @param dt Time step.
     * @param g Gravitational constant scale.
     * @param softening Softening factor (distance squared added).
     */
    fun simulate(dt: Float = Config.DT, g: Float = Config.G, softening: Float = Config.SOFTENING) {
        if (count <= 0) return
        glUseProgram(computeProg)
        glUniform1f(uDt, dt)
        glUniform1f(uSoft, softening * softening)
        glUniform1f(uG, g)
        glUniform1ui(uCountC, count)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)
        val groups = (count + Config.WORK_GROUP_SIZE - 1) / Config.WORK_GROUP_SIZE
        glDispatchCompute(groups, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        glUseProgram(0)
    }

    /**
     * Renders all bodies as points with per-vertex size and color derived from mass and speed.
     *
     * @param viewportW Viewport width in pixels.
     * @param viewportH Viewport height in pixels.
     * @param camAngle Camera yaw angle in radians.
     * @param center Center of mass to focus the camera on.
     */
    fun render(viewportW: Int, viewportH: Int, camAngle: Float, center: FloatArray) {
        if (count <= 0) return
        glUseProgram(renderProg)
        glBindVertexArray(vao)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)
        glUniform2f(uViewport, viewportW.toFloat(), viewportH.toFloat())
        glUniform1f(uPointBase, Config.POINT_SIZE)
        glUniform1f(uMassScale, Config.MASS_POINT_SCALE)
        glUniform1f(uCamAngle, camAngle)
        glUniform3f(uCenter, center[0], center[1], center[2])
        glEnable(GL_PROGRAM_POINT_SIZE)
        glUniform1f(uCamPitch, 0.2617994f)
        glUniform1f(uSpeedScale, 1f / 10_000f)
        glDrawArrays(GL_POINTS, 0, count)
        glBindVertexArray(0)
        glUseProgram(0)
    }

    /**
     * Releases all GPU resources.
     */
    override fun close() {
        glDeleteProgram(renderProg)
        glDeleteProgram(computeProg)
        glDeleteVertexArrays(vao)
        glDeleteBuffers(ssbo)
    }
}

/**
 * Generates a 2D disk of bodies with approximate tangential velocities.
 *
 * @param n Number of bodies.
 * @param w Width for positioning.
 * @param h Height for positioning.
 * @return List of bodies.
 */
private fun generateDisk(n: Int, w: Int, h: Int): List<Body> {
    val cx = w * 0.5f
    val cy = h * 0.5f
    val rMax = min(w, h) * 0.45f
    val rnd = Random(1)
    val out = ArrayList<Body>(n)
    for (i in 0 until n) {
        val r = rMax * sqrt(rnd.nextFloat())
        val a = rnd.nextFloat() * (2f * Math.PI).toFloat()
        val x = cx + r * cos(a)
        val y = cy + r * sin(a)
        val z = 1.0f + rnd.nextFloat() * 10.0f
        val v = 50f / max(10f, r)
        val vx = -v * sin(a)
        val vy =  v * cos(a)
        val m = 1.0f + rnd.nextFloat() * 2.0f
        out += Body(x, y, z, vx, vy, 0f, m)
    }
    return out
}

/**
 * Generates a 3D spherical volume distribution with tangential velocities and a central massive body.
 *
 * @param n Number of satellite bodies (approximate; one extra central mass is added).
 * @param w Width reference for positioning.
 * @param h Height reference for positioning.
 * @return List of bodies including a central massive body.
 */
private fun generateSphere(n: Int, w: Int, h: Int): List<Body> {
    fun cross(ax: Float, ay: Float, az: Float, bx: Float, by: Float, bz: Float): Triple<Float, Float, Float> =
        Triple(ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)

    fun norm(x: Float, y: Float, z: Float): Triple<Float, Float, Float> {
        val len = sqrt(x * x + y * y + z * z).coerceAtLeast(1e-8f)
        return Triple(x / len, y / len, z / len)
    }

    val cx = w * 0.5f
    val cy = h * 0.5f
    val cz = min(w, h) * 0.5f
    val rMax = min(w, h) * 0.45f
    val rnd = Random(1)
    val out = ArrayList<Body>(n)

    for (i in 0 until n) {
        val r = rMax * cbrt(rnd.nextFloat().toDouble()).toFloat()
        val z = rnd.nextFloat() * 2f - 1f
        val phi = rnd.nextFloat() * (2f * Math.PI).toFloat()
        val s = sqrt(max(0f, 1f - z * z))
        val rx = s * cos(phi)
        val ry = s * sin(phi)
        val rz = z
        val x = cx + r * rx
        val y = cy + r * ry
        val zPos = cz + r * rz
        val speed = 300_000f / max(10f, r)
        val az = 0f
        val ax = if (abs(rz) > 0.99f) 1f else 0f
        val ay = if (abs(rz) > 0.99f) 0f else 1f
        val (tx0, ty0, tz0) = cross(rx, ry, rz, ax, ay, az)
        val (tx, ty, tz) = norm(tx0, ty0, tz0)
        val vx = tx * speed
        val vy = ty * speed
        val vz = tz * speed
        val m = 1.0f
        out += Body(x, y, zPos, vx, vy, vz, m)
    }
    return out + Body(cx, cy, cz, 0f, 0f, 0f, 5_000_000f)
}

/**
 * Generates a rotationally supported disk galaxy with optional bar-like perturbation and jitter.
 *
 * The first body is the central mass; the rest are satellites whose velocities are assigned
 * to approximate circular motion from the enclosed mass profile.
 *
 * @param nTotal Total number of bodies including the central mass.
 * @param epsM2 Bar-like m=2 perturbation amplitude (dimensionless).
 * @param phi0 Phase of the m=2 perturbation in radians.
 * @param barTaperR Gaussian taper radius for the bar influence; null to auto.
 * @param radialScale Exponential scale length for radial distribution; null to auto.
 * @param speedJitter Fractional speed noise around circular speed.
 * @param radialJitter Fractional radial velocity jitter relative to circular speed.
 * @param clockwise Rotation direction.
 * @param rng Random generator.
 * @param vx Global X velocity offset.
 * @param vy Global Y velocity offset.
 * @param x Center X coordinate.
 * @param y Center Y coordinate.
 * @param r Maximum disk radius.
 * @param minR Minimum radius clamp.
 * @param centralMass Mass of the central body.
 * @param totalSatelliteMass Total mass distributed among satellites.
 * @return Mutable list of bodies (central + satellites).
 */
fun makeGalaxyDisk(
    nTotal: Int,
    epsM2: Double = 0.03,
    phi0: Double = 0.0,
    barTaperR: Double? = null,
    radialScale: Double? = null,
    speedJitter: Double = 0.01,
    radialJitter: Double = 0.0,
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

    bodies += Body(cx.toFloat(), cy.toFloat(), 0f, vx.toFloat(), vy.toFloat(), 0f, centralMass.toFloat())

    val mSat = if (sats > 0) totalSatelliteMass / sats else 0.0
    val Rd = radialScale ?: (rMax / 3.0)
    val taperR = barTaperR ?: (rMax * 0.6)

    fun sampleExpRadius(): Double {
        val u = rng.nextDouble()
        val A = exp(-(rMax - minR) / Rd)
        val t = 1 - u * (1 - A)
        return minR - Rd * ln(t)
    }

    repeat(sats) {
        val R = sampleExpRadius().coerceIn(minR, rMax)
        val theta = rng.nextDouble() * 2.0 * Math.PI
        val taper = exp(- (R / taperR) * (R / taperR))
        val R2 = R * (1.0 + epsM2 * cos(2.0 * (theta - phi0)) * taper)
        val px = cx + R2 * cos(theta)
        val py = cy + R2 * sin(theta)
        bodies += Body(px.toFloat(), py.toFloat(), 0f, 0f, 0f, 0f, mSat.toFloat())
    }

    data class RIdx(val i: Int, val r: Double)
    val sorted = bodies.mapIndexed { i, b ->
        RIdx(i, hypot(b.x.toDouble() - cx, b.y.toDouble() - cy))
    }.sortedBy { it.r }

    var acc = 0.0
    val Menc = DoubleArray(bodies.size)
    for (ri in sorted) { acc += bodies[ri.i].m.toDouble(); Menc[ri.i] = acc }

    for (i in 1 until bodies.size) {
        val b = bodies[i]
        val dx = b.x.toDouble() - cx
        val dy = b.y.toDouble() - cy
        val R = max(1e-6, hypot(dx, dy))
        val vCirc = sqrt(Config.G.toDouble() * Menc[i] / R)
        val v = vCirc * (1.0 + (rng.nextDouble() - 0.5) * 2.0 * speedJitter)
        val (tx, ty) = if (clockwise) (dy / R) to (-dx / R) else (-dy / R) to (dx / R)
        var vx0 = tx * v
        var vy0 = ty * v
        if (radialJitter > 0.0) {
            val vr = (rng.nextDouble() - 0.5) * 2.0 * radialJitter * vCirc
            vx0 += (dx / R) * vr
            vy0 += (dy / R) * vr
        }
        b.vx = (vx0 + vx).toFloat()
        b.vy = (vy0 + vy).toFloat()
        b.vz = 0f
    }

    return bodies
}

/**
 * Application entry point. Initializes OpenGL, spawns bodies, runs the simulation loop, and renders frames.
 */
fun main() {
    GLFWErrorCallback.createPrint(System.err).set()
    if (!glfwInit()) error("GLFW init failed")
    glfwDefaultWindowHints()
    glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE)
    glfwWindowHint(GLFW_SAMPLES, 0)

    val window = glfwCreateWindow(Config.WIDTH, Config.HEIGHT, "GPU N-Body (SSBO render)", 0, 0)
    if (window == 0L) {
        glfwTerminate()
        error("Failed to create window")
    }
    glfwMakeContextCurrent(window)
    glfwSwapInterval(0)
    GL.createCapabilities()

    val N = 50_000
    val bodies = generateSphere(N, Config.WIDTH, Config.HEIGHT)

    var camAngle = 0.0f
    val camSpeed = 0.25f

    GpuNBodyRenderer(bodies.size, bodies).use { sim ->
        var paused = false

        glfwSetKeyCallback(window) { _, key, _, action, _ ->
            if (action == GLFW_PRESS) {
                when (key) {
                    GLFW_KEY_ESCAPE -> glfwSetWindowShouldClose(window, true)
                    GLFW_KEY_SPACE  -> paused = !paused
                }
            }
        }

        var lastTime = glfwGetTime()
        var accTime = 0.0
        var frames = 0
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents()

            val now = glfwGetTime()
            val dtFrame = (now - lastTime).toFloat()
            lastTime = now

            if (!paused) {
                sim.simulate(Config.DT)
                camAngle += camSpeed * dtFrame
            }

            val center = sim.computeCenterOfMass()

            val bg = if (Config.BACKGROUND_DARK) 0.05f else 1.0f
            glViewport(0, 0, Config.WIDTH, Config.HEIGHT)
            glClearColor(bg, bg, bg, 1f)
            glClear(GL_COLOR_BUFFER_BIT)

            sim.render(Config.WIDTH, Config.HEIGHT, camAngle, center)

            glfwSwapBuffers(window)

            frames++
            accTime += (glfwGetTime() - now)
            if (accTime >= 1.0) {
                glfwSetWindowTitle(window, "GPU N-Body (SSBO render)  |  ${frames} FPS  |  N=$N")
                frames = 0
                accTime = 0.0
            }
        }
    }

    glfwMakeContextCurrent(0)
    glfwDestroyWindow(window)
    glfwTerminate()
    glfwSetErrorCallback(null)?.free()
}