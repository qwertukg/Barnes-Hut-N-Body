package gpu

import org.lwjgl.glfw.GLFW.*
import org.lwjgl.glfw.GLFWErrorCallback
import org.lwjgl.opengl.GL
import org.lwjgl.opengl.GL46.*
import org.lwjgl.system.MemoryUtil.*
import java.io.File
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Global configuration for windowing, physics, rendering, camera,
 * and data input (NEO catalog) used by the N-body demo.
 */
private object Config {

    // ---- Window / GL context ------------------------------------------------

    /** Window width in pixels for the GLFW context. */
    const val WIDTH: Int = 3440

    /** Window height in pixels for the GLFW context. */
    const val HEIGHT: Int = 1440


    // ---- Compute / workgroup ------------------------------------------------

    /**
     * Local workgroup size in X dimension for the compute shader.
     *
     * Must match `layout(local_size_x = WORK_GROUP_SIZE)` in the GLSL code.
     */
    const val WORK_GROUP_SIZE: Int = 256


    // ---- Physics parameters -------------------------------------------------

    /**
     * Gravitational constant scale used by the compute shader.
     *
     * Units are internal to the simulation and chosen to give stable orbits.
     */
    const val G: Float = 80.0f

    /**
     * Fixed physics timestep for the integrator (seconds in simulation units).
     */
    const val DT: Float = 0.01f

    /**
     * Plummer-like softening length.
     *
     * In the compute shader, `uSoftening` is used as `soft^2` added to r^2,
     * so this value is effectively a length scale.
     */
    const val SOFTENING: Float = 1.0f


    // ---- Legacy rendering parameters (не используются, но оставлены) -------

    /** Base OpenGL point size (in pixels). */
    const val POINT_SIZE: Float = 5f

    /** Additional point size contribution per unit of mass (unused). */
    const val MASS_POINT_SCALE: Float = 0.0001f

    /** Whether to use a dark background (true) or white background (false). */
    const val BACKGROUND_DARK: Boolean = true

    /** Old speed→color scale (не используется). */
    const val SPEED_COLOR_SCALE: Float = 1f / 10_000f


    // ---- Mass / scale model -------------------------------------------------

    /** Mass of the Earth in simulation units. */
    const val EARTH_MASS: Float = 1f

    /** Ratio of the Sun mass to Earth mass. */
    const val SUN_EARTH_MASS_RATIO: Float = 333_000f

    /** Mass of the Sun in simulation units. */
    const val SUN_MASS: Float = EARTH_MASS * SUN_EARTH_MASS_RATIO

    /**
     * Fraction of the minimum screen dimension used as Earth's orbital radius.
     *
     * If min(W, H) is the smaller screen size in pixels, then
     * Earth orbit radius = EARTH_ORBIT_FRACTION * min(W, H).
     */
    const val EARTH_ORBIT_FRACTION: Float = 0.25f


    // ---- Camera parameters --------------------------------------------------

    /**
     * Initial camera pitch angle in radians (rotation around X axis).
     */
    const val CAMERA_PITCH_RAD: Float = -0.2617994f // ~15 degrees

    /**
     * Initial distance from camera to the center of the system.
     */
    const val CAMERA_DISTANCE: Float = 3000f

    /** Min/max zoom distance. */
    const val CAMERA_DISTANCE_MIN: Float = 800f
    const val CAMERA_DISTANCE_MAX: Float = 3_000f

    /** Yaw (A/D) and pitch (W/S) speeds, rad/sec. */
    const val CAMERA_YAW_SPEED: Float = 0.7f
    const val CAMERA_PITCH_SPEED: Float = 0.7f

    /** Pitch limits, radians. */
    const val CAMERA_PITCH_MIN_RAD: Float = -1.2f   // ~ -69°
    const val CAMERA_PITCH_MAX_RAD: Float = 0.3f    // ~ +17°

    /** Zoom factor per scroll step (fraction of distance). */
    const val CAMERA_ZOOM_FACTOR_PER_SCROLL: Float = 0.1f

    /** Vertical FOV (radians). */
    const val CAMERA_FOV_Y_RAD: Float = 0.7853982f // ~45 degrees

    /** Near and far clipping planes for the perspective camera. */
    const val CAMERA_NEAR: Float = 50f
    const val CAMERA_FAR: Float = 10_000f

    /** Old auto-rotate speed (now unused, но оставлен). */
    const val CAMERA_SPEED: Float = 0.0f


    // ---- Per-body-type radii (world units, same scale as positions) --------

    /** Sun radius in simulation units. */
    const val SUN_RADIUS: Float = 40f

    /** Earth radius in simulation units. */
    const val EARTH_RADIUS: Float = 10f

    /** Asteroid radius in simulation units. */
    const val ASTEROID_RADIUS: Float = 2f


    // ---- Per-body-type colors (RGB 0..1) -----------------------------------

    /** Sun color (yellow-ish). */
    val SUN_COLOR: FloatArray = floatArrayOf(1.0f, 0.9f, 0.1f)

    /** Earth color (green-ish). */
    val EARTH_COLOR: FloatArray = floatArrayOf(0.1f, 0.9f, 0.2f)

    /** Asteroid color (neutral gray/white). */
    val ASTEROID_COLOR: FloatArray = floatArrayOf(1.0f, 1.0f, 1.0f)


    // ---- Asteroid brightness falloff params --------------------------------

    /** Минимальная яркость астероидов (на дальнем пределе). */
    const val ASTEROID_MIN_BRIGHTNESS: Float = 0.1f

    /** Максимальная яркость астероидов (рядом с камерой). */
    const val ASTEROID_MAX_BRIGHTNESS: Float = 1.0f

    /** Степень кривой яркости: 1.0 — линейно, >1 — сильнее гаснут вдали. */
    const val ASTEROID_BRIGHTNESS_POWER: Float = 3.0f


    // ---- NEO catalog parameters --------------------------------------------

    /** Path to the ESA NEO Keplerian elements catalog file (neo_kc.cat). */
    const val NEO_FILE_PATH: String = "src/main/resources/neo_kc.cat"

    /**
     * 1-based index of the first line to read from the NEO catalog.
     *
     * All previous lines are treated as header/metadata.
     */
    const val NEO_START_LINE: Int = 6

    /** Количество сэмплов для MSAA (0 = выключен, 4 или 8 = включен). */
    const val MSAA_SAMPLES: Int = 4

}


/**
 * Body representation used on the CPU side and in the GPU SSBO.
 */
data class Body(
    var x: Float, var y: Float, var z: Float,
    var vx: Float, var vy: Float, var vz: Float,
    var m: Float
)

/**
 * Builds and links the compute shader program used for the N-body simulation.
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
 */
private fun buildRenderProgram(): Int {
    val vs = """
#version 460 core
struct Body { vec4 posMass; vec4 velPad; };
layout(std430, binding = 0) buffer BodyBuffer { Body bodies[]; };

uniform vec2  uViewport;

// camera
uniform float uCamAngle;
uniform float uCamPitch;
uniform float uCamRadius;
uniform vec3  uCenter;
uniform float uFovY;
uniform float uNear;
uniform float uFar;

// body classification / radii
uniform float uSunMass;
uniform float uSunRadius;
uniform float uEarthRadius;
uniform float uAsteroidRadius;

// asteroid brightness params
uniform float uAsteroidMinBrightness;
uniform float uAsteroidMaxBrightness;
uniform float uAsteroidBrightnessPower;

out float vMass;
out float vBrightness;

void main(){
    int id = gl_VertexID;
    vec3 worldPos = bodies[id].posMass.xyz;
    float m       = bodies[id].posMass.w;

    // --- camera position (spherical around center) ---
    float yaw   = uCamAngle;
    float pitch = uCamPitch;

    float cy = cos(yaw);
    float sy = sin(yaw);
    float cp = cos(pitch);
    float sp = sin(pitch);

    // forward from camera to center
    vec3 forward = normalize(vec3(cy * cp, sp, sy * cp));

    // camera eye position
    vec3 eye = uCenter - forward * uCamRadius;

    // view basis (right, up, -forward)
    vec3 f  = normalize(uCenter - eye);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 s  = normalize(cross(f, up));
    vec3 u  = cross(s, f);

    // world -> view
    vec3 rel  = worldPos - eye;
    vec3 view;
    view.x = dot(rel, s);
    view.y = dot(rel, u);
    view.z = dot(rel, -f);  // negative in front of camera

    // расстояние до камеры в view-пространстве
    float dist = length(view);

    // --- perspective projection ---
    float aspect = uViewport.x / uViewport.y;
    float fScale = 1.0 / tan(uFovY * 0.5);

    float n = uNear;
    float fa = uFar;

    mat4 proj = mat4(
        vec4(fScale / aspect, 0.0,  0.0,                      0.0),
        vec4(0.0,             fScale, 0.0,                    0.0),
        vec4(0.0,             0.0,   (fa + n) / (n - fa),    -1.0),
        vec4(0.0,             0.0,   (2.0 * fa * n) / (n - fa), 0.0)
    );

    vec4 clip = proj * vec4(view, 1.0);
    gl_Position = clip;

    // --- choose physical radius by body type ---
    float radius;
    if (m > 0.5 * uSunMass) {
        // Sun
        radius = uSunRadius;
    } else if (m > 0.0) {
        // Earth-like mass
        radius = uEarthRadius;
    } else {
        // Asteroids (massless test particles)
        radius = uAsteroidRadius;
    }

    // perspective-correct size in pixels: size ∝ radius / distance
    float projScale = (uViewport.y * 0.5) / tan(uFovY * 0.5);
    float sizePx = radius * projScale / max(dist, 1e-3);

    gl_PointSize = max(sizePx, 1.0);
    vMass = m;

    // яркость для астероидов: ближе -> светлее, дальше -> темнее
    float t = (dist - uNear) / (uFar - uNear);
    t = clamp(t, 0.0, 1.0);

    float w = pow(1.0 - t, uAsteroidBrightnessPower);
    vBrightness = mix(uAsteroidMinBrightness, uAsteroidMaxBrightness, w);
}
""".trimIndent()

    val fs = """
#version 460 core
in float vMass;
in float vBrightness;
out vec4 fragColor;

uniform float uSunMass;
uniform vec3  uSunColor;
uniform vec3  uEarthColor;
uniform vec3  uAsteroidColor;

void main(){
    // круглый спрайт
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    if (dot(c,c) > 1.0) discard;

    vec3 color;
    if (vMass > 0.5 * uSunMass) {
        color = uSunColor;
    } else if (vMass > 0.0) {
        color = uEarthColor;
    } else {
        color = uAsteroidColor * vBrightness;
    }

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
 * GPU N-body engine wrapper that manages SSBO storage, simulation,
 * and rendering of all bodies.
 */
private class GpuNBodyRenderer(
    private var count: Int,
    bodies: List<Body>
) : AutoCloseable {

    private val ssbo = glGenBuffers()
    private val vao = glGenVertexArrays()
    private val computeProg = buildComputeProgram()
    private val renderProg = buildRenderProgram()

    // compute uniforms
    private val uDt      = glGetUniformLocation(computeProg, "uDt")
    private val uSoft    = glGetUniformLocation(computeProg, "uSoftening")
    private val uG       = glGetUniformLocation(computeProg, "uG")
    private val uCountC  = glGetUniformLocation(computeProg, "uCount")

    // render uniforms
    private val uViewport          = glGetUniformLocation(renderProg, "uViewport")
    private val uCamAngle          = glGetUniformLocation(renderProg, "uCamAngle")
    private val uCamPitch          = glGetUniformLocation(renderProg, "uCamPitch")
    private val uCamRadius         = glGetUniformLocation(renderProg, "uCamRadius")
    private val uCenter            = glGetUniformLocation(renderProg, "uCenter")
    private val uFovY              = glGetUniformLocation(renderProg, "uFovY")
    private val uNear              = glGetUniformLocation(renderProg, "uNear")
    private val uFar               = glGetUniformLocation(renderProg, "uFar")
    private val uSunMassR          = glGetUniformLocation(renderProg, "uSunMass")
    private val uSunRadius         = glGetUniformLocation(renderProg, "uSunRadius")
    private val uEarthRadius       = glGetUniformLocation(renderProg, "uEarthRadius")
    private val uAsteroidRadius    = glGetUniformLocation(renderProg, "uAsteroidRadius")
    private val uSunColor          = glGetUniformLocation(renderProg, "uSunColor")
    private val uEarthColor        = glGetUniformLocation(renderProg, "uEarthColor")
    private val uAsteroidColor     = glGetUniformLocation(renderProg, "uAsteroidColor")
    private val uAstMinBrightness  = glGetUniformLocation(renderProg, "uAsteroidMinBrightness")
    private val uAstMaxBrightness  = glGetUniformLocation(renderProg, "uAsteroidMaxBrightness")
    private val uAstBrightnessPow  = glGetUniformLocation(renderProg, "uAsteroidBrightnessPower")

    // старые юниформы (можно игнорировать, даже если -1)
    private val uPointBase         = glGetUniformLocation(renderProg, "uPointBase")
    private val uMassScale         = glGetUniformLocation(renderProg, "uMassScale")
    private val uSpeedScale        = glGetUniformLocation(renderProg, "uSpeedScale")

    private var capacityBytes: Long = 0L

    init {
        glBindVertexArray(vao)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, 0L, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        ensureCapacity(count)
        uploadBodies(bodies)
    }

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
        } finally {
            memFree(buf)
        }
    }

    fun computeCenterOfMass(): FloatArray {
        if (count == 0) return floatArrayOf(0f, 0f, 0f)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        val floatsPerBody = 8
        val totalFloats = count * floatsPerBody
        val buf = memAllocFloat(totalFloats)
        try {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo)
            glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0L, buf)
            var sx = 0.0
            var sy = 0.0
            var sz = 0.0
            var sm = 0.0
            for (i in 0 until count) {
                val base = i * floatsPerBody
                val x = buf.get(base + 0).toDouble()
                val y = buf.get(base + 1).toDouble()
                val z = buf.get(base + 2).toDouble()
                val m = buf.get(base + 3).toDouble()
                sx += x * m
                sy += y * m
                sz += z * m
                sm += m
            }
            if (sm == 0.0) return floatArrayOf(0f, 0f, 0f)
            return floatArrayOf(
                (sx / sm).toFloat(),
                (sy / sm).toFloat(),
                (sz / sm).toFloat()
            )
        } finally {
            memFree(buf)
        }
    }

    fun simulate(
        dt: Float = Config.DT,
        g: Float = Config.G,
        softening: Float = Config.SOFTENING
    ) {
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

    fun render(
        viewportW: Int,
        viewportH: Int,
        camAngle: Float,
        camPitch: Float,
        camRadius: Float,
        center: FloatArray
    ) {
        if (count <= 0) return
        glUseProgram(renderProg)
        glBindVertexArray(vao)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ssbo)

        glUniform2f(uViewport, viewportW.toFloat(), viewportH.toFloat())
        glUniform1f(uCamAngle, camAngle)
        glUniform1f(uCamPitch, camPitch)
        glUniform1f(uCamRadius, camRadius)
        glUniform3f(uCenter, center[0], center[1], center[2])
        glUniform1f(uFovY, Config.CAMERA_FOV_Y_RAD)
        glUniform1f(uNear, Config.CAMERA_NEAR)
        glUniform1f(uFar, Config.CAMERA_FAR)

        glUniform1f(uSunMassR, Config.SUN_MASS)
        glUniform1f(uSunRadius, Config.SUN_RADIUS)
        glUniform1f(uEarthRadius, Config.EARTH_RADIUS)
        glUniform1f(uAsteroidRadius, Config.ASTEROID_RADIUS)

        Config.SUN_COLOR.also { glUniform3f(uSunColor, it[0], it[1], it[2]) }
        Config.EARTH_COLOR.also { glUniform3f(uEarthColor, it[0], it[1], it[2]) }
        Config.ASTEROID_COLOR.also { glUniform3f(uAsteroidColor, it[0], it[1], it[2]) }

        glUniform1f(uAstMinBrightness, Config.ASTEROID_MIN_BRIGHTNESS)
        glUniform1f(uAstMaxBrightness, Config.ASTEROID_MAX_BRIGHTNESS)
        glUniform1f(uAstBrightnessPow, Config.ASTEROID_BRIGHTNESS_POWER)

        glUniform1f(uPointBase, Config.POINT_SIZE)
        glUniform1f(uMassScale, Config.MASS_POINT_SCALE)
        glUniform1f(uSpeedScale, Config.SPEED_COLOR_SCALE)

        glEnable(GL_PROGRAM_POINT_SIZE)
        glEnable(GL_DEPTH_TEST)

        glDrawArrays(GL_POINTS, 0, count)
        glBindVertexArray(0)
        glUseProgram(0)
    }

    override fun close() {
        glDeleteProgram(renderProg)
        glDeleteProgram(computeProg)
        glDeleteVertexArrays(vao)
        glDeleteBuffers(ssbo)
    }
}

/**
 * Creates the base two-body system: a massive Sun and an Earth-like planet.
 */
private fun createBase(): List<Body> {
    val w = Config.WIDTH
    val h = Config.HEIGHT

    val cx = w * 0.5f
    val cy = h * 0.5f
    val cz = 0f

    val r = (min(w, h) * Config.EARTH_ORBIT_FRACTION).toFloat()

    val earthMass = Config.EARTH_MASS
    val sunMass = Config.SUN_MASS

    val earthSpeed = sqrt((Config.G * sunMass / r).toDouble()).toFloat()

    val sunVz = -earthSpeed * (earthMass / sunMass)

    val sun = Body(
        x = cx,
        y = cy,
        z = cz,
        vx = 0f,
        vy = 0f,
        vz = sunVz,
        m = sunMass
    )

    val earth = Body(
        x = cx + r,
        y = cy,
        z = cz,
        vx = 0f,
        vy = 0f,
        vz = earthSpeed,
        m = earthMass
    )

    return listOf(sun, earth)
}

private fun degToRad(d: Double): Double = d * PI / 180.0

private fun solveKeplerE(M: Double, e: Double, iters: Int = 16): Double {
    var E = M
    repeat(iters) {
        val f = E - e * sin(E) - M
        val fp = 1.0 - e * cos(E)
        E -= f / fp
    }
    return E
}

private fun createAsteroidFromElements(
    aAu: Double,
    e: Double,
    iDeg: Double,
    longNodeDeg: Double,
    argPericDeg: Double,
    meanAnomalyDeg: Double,
    sunMass: Float = Config.SUN_MASS
): Body {
    val rEarthSim = (min(Config.WIDTH, Config.HEIGHT) * Config.EARTH_ORBIT_FRACTION).toDouble()
    val aSim = aAu * rEarthSim

    val i = degToRad(iDeg)
    val Om = degToRad(longNodeDeg)
    val w = degToRad(argPericDeg)
    val M = degToRad(meanAnomalyDeg)

    val E = solveKeplerE(M, e)
    val cosE = cos(E)
    val sinE = sin(E)
    val sqrt1me2 = sqrt(1.0 - e * e)

    val r = aSim * (1.0 - e * cosE)

    val xP = aSim * (cosE - e)
    val yP = aSim * (sqrt1me2 * sinE)

    val mu = Config.G.toDouble() * sunMass.toDouble()

    val factor = sqrt(mu * aSim) / r
    val vxP = -factor * sinE
    val vyP = factor * sqrt1me2 * cosE

    val cO = cos(Om); val sO = sin(Om)
    val ci = cos(i);  val si = sin(i)
    val cw = cos(w);  val sw = sin(w)

    val r11 =  cO * cw - sO * sw * ci
    val r12 = -cO * sw - sO * cw * ci
    val r21 =  sO * cw + cO * sw * ci
    val r22 = -sO * sw + cO * cw * ci
    val r31 =  sw * si
    val r32 =  cw * si

    val xE = r11 * xP + r12 * yP
    val yE = r21 * xP + r22 * yP
    val zE = r31 * xP + r32 * yP

    val vxE = r11 * vxP + r12 * vyP
    val vyE = r21 * vxP + r22 * vyP
    val vzE = r31 * vxP + r32 * vyP

    val xSim = xE
    val ySim = zE
    val zSim = yE
    val vxSim = vxE
    val vySim = vzE
    val vzSim = vyE

    val cx = Config.WIDTH * 0.5f
    val cy = Config.HEIGHT * 0.5f
    val cz = 0f

    val mAst = 0f

    return Body(
        x = (cx + xSim).toFloat(),
        y = (cy + ySim).toFloat(),
        z = (cz + zSim).toFloat(),
        vx = vxSim.toFloat(),
        vy = vySim.toFloat(),
        vz = vzSim.toFloat(),
        m = mAst
    )
}

private fun createAsteroidFromNeoLine(line: String): Body {
    val trimmed = line.trim()
    require(!trimmed.startsWith("!")) { "Header/comment line passed to createAsteroidFromNeoLine" }
    val p = trimmed.split(Regex("\\s+"))
    require(p.size >= 11) { "Expected at least 11 columns, got: ${p.size}" }

    val aAu = p[2].toDouble()
    val e = p[3].toDouble()
    val iDeg = p[4].toDouble()
    val longNodeDeg = p[5].toDouble()
    val argPericDeg = p[6].toDouble()
    val meanAnomalyDeg = p[7].toDouble()

    return createAsteroidFromElements(
        aAu = aAu,
        e = e,
        iDeg = iDeg,
        longNodeDeg = longNodeDeg,
        argPericDeg = argPericDeg,
        meanAnomalyDeg = meanAnomalyDeg
    )
}

fun readLinesFrom(path: String, startLine: Int = Config.NEO_START_LINE): List<String> {
    require(startLine >= 1) { "startLine must be >= 1" }

    val file = File(path)
    val result = mutableListOf<String>()

    file.bufferedReader().use { reader ->
        var current = 1
        while (true) {
            val line = reader.readLine() ?: break
            if (current >= startLine) {
                result += line
            }
            current++
        }
    }
    return result
}

fun createWorld(): List<Body> {
    val base = createBase()
    val dataLines = readLinesFrom(Config.NEO_FILE_PATH)
    val asteroids = dataLines
        .filter { it.isNotBlank() && !it.trim().startsWith("!") }
        .map { createAsteroidFromNeoLine(it) }
    return base + asteroids
}

/**
 * Application entry point.
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
    glfwWindowHint(GLFW_SAMPLES, Config.MSAA_SAMPLES)
    glfwWindowHint(GLFW_DEPTH_BITS, 24)

    // определяем монитор и режим
    val monitor = glfwGetPrimaryMonitor()
    val videoMode = glfwGetVideoMode(monitor) ?: error("No video mode for primary monitor")

    val window = glfwCreateWindow(
        videoMode.width(),
        videoMode.height(),
        "GPU N-Body (SSBO render)",
        monitor,
        0
    )
    if (window == 0L) {
        glfwTerminate()
        error("Failed to create window")
    }

    glfwMakeContextCurrent(window)
    glfwSwapInterval(0)
    GL.createCapabilities()

    if (Config.MSAA_SAMPLES > 0) {
        glEnable(GL_MULTISAMPLE)
    }

    glEnable(GL_DEPTH_TEST)

    val bodies = createWorld()
    val N = bodies.size

    // состояние камеры
    var camAngle = 0.0f
    var camPitch = Config.CAMERA_PITCH_RAD
    var camRadius = Config.CAMERA_DISTANCE

    var paused = false

    glfwSetKeyCallback(window) { _, key, _, action, _ ->
        if (action == GLFW_PRESS) {
            when (key) {
                GLFW_KEY_ESCAPE -> glfwSetWindowShouldClose(window, true)
                GLFW_KEY_SPACE  -> paused = !paused
            }
        }
    }

    // Зум на колёсико
    glfwSetScrollCallback(window) { _, _, yoffset ->
        if (yoffset != 0.0) {
            val factor = 1.0f - Config.CAMERA_ZOOM_FACTOR_PER_SCROLL * yoffset.toFloat()
            camRadius *= factor
            camRadius = camRadius.coerceIn(
                Config.CAMERA_DISTANCE_MIN,
                Config.CAMERA_DISTANCE_MAX
            )
        }
    }

    var lastTime = glfwGetTime()
    var accTime = 0.0
    var frames = 0

    GpuNBodyRenderer(bodies.size, bodies).use { sim ->
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents()

            val now = glfwGetTime()
            val dtFrame = (now - lastTime).toFloat()
            lastTime = now

            // вращение камеры WASD
            if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
                camAngle += Config.CAMERA_YAW_SPEED * dtFrame
            }
            if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
                camAngle -= Config.CAMERA_YAW_SPEED * dtFrame
            }
            if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
                camPitch += Config.CAMERA_PITCH_SPEED * dtFrame
            }
            if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
                camPitch -= Config.CAMERA_PITCH_SPEED * dtFrame
            }
            camPitch = camPitch.coerceIn(
                Config.CAMERA_PITCH_MIN_RAD,
                Config.CAMERA_PITCH_MAX_RAD
            )

            if (!paused) {
                // фиксированный шаг как раньше — не трогаем гравитацию
                sim.simulate(Config.DT)
            }

            val center = sim.computeCenterOfMass()

            val bg = if (Config.BACKGROUND_DARK) 0.05f else 1.0f
            glViewport(0, 0, Config.WIDTH, Config.HEIGHT)
            glClearColor(bg, bg, bg, 1f)
            glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)

            sim.render(
                Config.WIDTH,
                Config.HEIGHT,
                camAngle,
                camPitch,
                camRadius,
                center
            )

            glfwSwapBuffers(window)

            frames++
            accTime += (glfwGetTime() - now)
            if (accTime >= 1.0) {
                glfwSetWindowTitle(
                    window,
                    "GPU N-Body (SSBO render)  |  ${frames} FPS  |  N=$N"
                )
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
