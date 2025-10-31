@file:JvmName("GpuBarnesHut_FullGPU")
package gpu

import org.lwjgl.glfw.GLFW.*
import org.lwjgl.glfw.GLFWErrorCallback
import org.lwjgl.opengl.GL
import org.lwjgl.opengl.GL46.*
import org.lwjgl.system.MemoryUtil.NULL
import kotlin.math.max

/** Полностью-GPU Barnes–Hut (GL 4.6, GLSL 460). CPU только диспатчит шейдеры и рисует. */
object Config {
    // Экран
    const val WIDTH  = 1920
    const val HEIGHT = 1080

    // Размер воркгруппы (256/512/1024 — см. лимиты драйвера)
    const val WORKGROUP = 256

    // Физика
    const val G      = 80.0f
    const val DT     = 0.005f
    const val SOFT   = 1.0f
    const val THETA  = 0.5f

    // Morton / Quadtree (2 бита Morton = 1 уровень)
    const val MAX_DEPTH = 16
    const val GRID_W    = 1 shl MAX_DEPTH
    const val GRID_H    = 1 shl MAX_DEPTH

    // Merge
    const val MERGE_MIN_DIST = 4.0f
    const val MERGE_MAX_MASS = 4000.0f
    const val GRID_CELLS  = 2048                // степень 2
    const val GRID_BUCKET = 32

    // Radix sort (4-bit): 8 проходов по 32-битным ключам
    const val RADIX_DIGITS = 16

    // Глубина стека обхода узлов на тело
    const val STACK_CAP = 256
}

/* ============================== OpenGL helpers ============================== */

private fun compileShader(type: Int, src: String): Int {
    val id = glCreateShader(type)
    glShaderSource(id, src)
    glCompileShader(id)
    if (glGetShaderi(id, GL_COMPILE_STATUS) == GL_FALSE) {
        val log = glGetShaderInfoLog(id)
        glDeleteShader(id)
        error("Shader compile error:\n$log")
    }
    return id
}
private fun linkProgram(vararg shaders: Int): Int {
    val p = glCreateProgram()
    shaders.forEach { glAttachShader(p, it) }
    glLinkProgram(p)
    if (glGetProgrami(p, GL_LINK_STATUS) == GL_FALSE) {
        val log = glGetProgramInfoLog(p)
        shaders.forEach { try { glDeleteShader(it) } catch (_: Throwable) {} }
        glDeleteProgram(p)
        error("Program link error:\n$log")
    }
    shaders.forEach { glDetachShader(p, it); glDeleteShader(it) }
    return p
}
private fun createSSBO(binding: Int, bytes: Long, usage: Int = GL_DYNAMIC_DRAW): Int {
    val id = glGenBuffers()
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id)
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, usage)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, id)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return id
}
private fun ceilDiv(a: Int, b: Int) = (a + b - 1) / b
private fun maxOf(vararg v: Int) = v.max()

/* ============================== Main ============================== */

fun main() {
    GLFWErrorCallback.createPrint(System.err).set()
    check(glfwInit()) { "GLFW init failed" }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE)

    val win = glfwCreateWindow(Config.WIDTH, Config.HEIGHT, "Barnes–Hut GPU (GLSL 460)", NULL, NULL)
    check(win != NULL) { "Window create failed" }
    glfwMakeContextCurrent(win)
    glfwSwapInterval(0)
    GL.createCapabilities()

    try {
        glEnable(GL_DEBUG_OUTPUT)
        glDebugMessageCallback({ _, _, _, _, length, message, _ ->
            val msg = org.lwjgl.system.MemoryUtil.memUTF8(message, length)
            System.err.println("GL DEBUG: $msg")
        }, 0)
    } catch (_: Throwable) {}

    /* ------------------------ Параметры симуляции ------------------------ */

    // Кол-во тел (можешь менять, O(N log N)). Пройдет и 200k+ при нормальном GPU.
    val N = 200_000

    /* ----------------------------- Буферы ----------------------------- */
    val floatsPerBody = 6 // pos.xy, vel.xy, mass, pad
    val bodiesSSBO = createSSBO(0, N.toLong() * floatsPerBody * 4L)       // Bodies
    val keysSSBO   = createSSBO(1, N.toLong() * 4L)                        // morton
    val indexSSBO  = createSSBO(2, N.toLong() * 4L)                        // indexIn
    val keysTmp    = createSSBO(3, N.toLong() * 4L)                        // mortonT
    val idxTmp     = createSSBO(4, N.toLong() * 4L)                        // indexT

    // Узлы LBVH: внутренние [0..N-2], листья [N-1..2N-2]
    val nodeU32 = 16
    val nodesSSBO = createSSBO(5, (2L * N) * nodeU32 * 4L)

    // Стек обхода узлов: STACK_CAP uint на тело
    val stackSSBO = createSSBO(6, N.toLong() * Config.STACK_CAP * 4L)

    // Merge (hash grid) + dead flags
    val gridHeadsSSBO = createSSBO(7, Config.GRID_CELLS.toLong() * 4L)
    val gridListSSBO  = createSSBO(8, (Config.GRID_CELLS * Config.GRID_BUCKET).toLong() * 4L)
    val deadFlagsSSBO = createSSBO(9, N.toLong() * 4L) // uint 0/1

    // Ускорения ax, ay
    val accelSSBO = createSSBO(10, N.toLong() * 2L * 4L)

    // Radix sort промежуточные: counts, prefix, globals, globals-scan
    val blocks = ceilDiv(N, Config.WORKGROUP)
    val countsSSBO      = createSSBO(11, blocks.toLong() * Config.RADIX_DIGITS * 4L) // [blocks][16]
    val prefixSSBO      = createSSBO(12, blocks.toLong() * Config.RADIX_DIGITS * 4L) // [blocks][16]
    val globalsSSBO     = createSSBO(13, Config.RADIX_DIGITS.toLong() * 4L)          // [16]
    val globalsScanSSBO = createSSBO(14, Config.RADIX_DIGITS.toLong() * 4L)          // [16]

    // SSBO для корня дерева (uint rootIdx[1])
    val rootSSBO = createSSBO(15, 4L)

    /* ----------------------------- Шейдеры ----------------------------- */
    val csInitBodies   = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.initBodies))
    val csMorton       = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.morton))

    val csRadixCount   = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.radixCount))
    val csRadixPrefix  = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.radixPrefix))
    val csRadixScatter = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.radixScatter))

    val csReindexBodies = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.reindexBodies))
    val csBuildNodes    = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.buildNodesLBVH))
    val csMassLeaves    = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.massLeaves))
    val csMassLevels    = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.massReduceLevels))

    val csRootClear = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.rootClear))
    val csFindRoot  = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.findRoot))

    val csAccel    = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.accumulateForces))
    val csKickHalf = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.kickHalf))
    val csDrift    = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.drift))

    val csGridClear = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.gridClear))
    val csGridFill  = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.gridFill))
    val csMerge     = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.mergeBodies))
    val csCompact   = linkProgram(compileShader(GL_COMPUTE_SHADER, Shaders.compactDead))

    val vsRender = compileShader(GL_VERTEX_SHADER, Shaders.vsRender)
    val fsRender = compileShader(GL_FRAGMENT_SHADER, Shaders.fsRender)
    val progRender = linkProgram(vsRender, fsRender)

    val vao = glGenVertexArrays()
    glBindVertexArray(vao)

    /* ----------------------------- UBO ----------------------------- */
    val ubo = glGenBuffers()
    glBindBuffer(GL_UNIFORM_BUFFER, ubo)
    glBufferData(GL_UNIFORM_BUFFER, 64, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_UNIFORM_BUFFER, 0, ubo)
    fun updateUBO() {
        glBindBuffer(GL_UNIFORM_BUFFER, ubo)
        val bb = org.lwjgl.BufferUtils.createByteBuffer(64)
        bb.putFloat(Config.G)
        bb.putFloat(Config.DT)
        bb.putFloat(Config.SOFT)
        bb.putFloat(Config.THETA)
        bb.putInt(Config.WIDTH); bb.putInt(Config.HEIGHT)
        bb.putFloat(Config.MERGE_MIN_DIST)
        bb.putFloat(Config.MERGE_MAX_MASS)
        bb.flip()
        glBufferSubData(GL_UNIFORM_BUFFER, 0, bb)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)
    }
    updateUBO()

    /* ----------------------------- Инициализация тел ----------------------------- */
    glUseProgram(csInitBodies)
    glUniform1i(glGetUniformLocation(csInitBodies, "uCount"), N)
    glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT or GL_BUFFER_UPDATE_BARRIER_BIT)

    /* ============================= Рендер-цикл ============================= */
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents()

        // ====== a) Построение дерева @ t ======
        dispatchMortonSortBuildMass(N, blocks,
            csMorton, csRadixCount, csRadixPrefix, csRadixScatter,
            csReindexBodies, csBuildNodes, csMassLeaves, csMassLevels,
            csRootClear, csFindRoot
        )

        // ====== b) a(t) ======
        glUseProgram(csAccel)
        glUniform1i(glGetUniformLocation(csAccel, "uCount"), N)
        glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // ====== c) kick(dt/2) ======
        glUseProgram(csKickHalf)
        glUniform1i(glGetUniformLocation(csKickHalf, "uCount"), N)
        glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // ====== d) drift(dt) ======
        glUseProgram(csDrift)
        glUniform1i(glGetUniformLocation(csDrift, "uCount"), N)
        glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // ====== e) Построение дерева @ t+dt ======
        dispatchMortonSortBuildMass(N, blocks,
            csMorton, csRadixCount, csRadixPrefix, csRadixScatter,
            csReindexBodies, csBuildNodes, csMassLeaves, csMassLevels,
            csRootClear, csFindRoot
        )

        // ====== f) a(t+dt) ======
        glUseProgram(csAccel)
        glUniform1i(glGetUniformLocation(csAccel, "uCount"), N)
        glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // ====== g) kick(dt/2) ======
        glUseProgram(csKickHalf)
        glUniform1i(glGetUniformLocation(csKickHalf, "uCount"), N)
        glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // ====== h) MERGE (+compact) ======
        if (Config.MERGE_MIN_DIST > 0f) {
            glUseProgram(csGridClear)
            glDispatchCompute(
                ceilDiv(maxOf(Config.GRID_CELLS, Config.GRID_CELLS * Config.GRID_BUCKET), Config.WORKGROUP),
                1, 1
            )
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(csGridFill)
            glUniform1i(glGetUniformLocation(csGridFill, "uCount"), N)
            glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(csMerge)
            glUniform1i(glGetUniformLocation(csMerge, "uCount"), N)
            glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(csCompact)
            glUniform1i(glGetUniformLocation(csCompact, "uCount"), N)
            glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        }

        // ====== i) Рендер из SSBO ======
        glViewport(0, 0, Config.WIDTH, Config.HEIGHT)
        glClearColor(0f, 0f, 0f, 1f)
        glClear(GL_COLOR_BUFFER_BIT)
        glUseProgram(progRender)
        glUniform2f(glGetUniformLocation(progRender, "uScreen"), Config.WIDTH.toFloat(), Config.HEIGHT.toFloat())
        glDrawArrays(GL_POINTS, 0, N)
        glfwSwapBuffers(win)
    }

    glfwDestroyWindow(win)
    glfwTerminate()
}

/* ——— Morton→Radix→Reindex→BuildNodes→Mass→Root (вынесено в функцию) ——— */
private fun dispatchMortonSortBuildMass(
    N: Int, blocks: Int,
    csMorton: Int, csRadixCount: Int, csRadixPrefix: Int, csRadixScatter: Int,
    csReindexBodies: Int, csBuildNodes: Int, csMassLeaves: Int, csMassLevels: Int,
    csRootClear: Int, csFindRoot: Int
) {
    // Morton
    glUseProgram(csMorton)
    glUniform1i(glGetUniformLocation(csMorton, "uCount"), N)
    glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    // Radix (8 проходов по 4 бита)
    repeat(8) { pass ->
        // Count
        glUseProgram(csRadixCount)
        glUniform1i(glGetUniformLocation(csRadixCount, "uCount"), N)
        glUniform1i(glGetUniformLocation(csRadixCount, "uPass"), pass)
        glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // Prefix (по блокам) — всё на GPU
        glUseProgram(csRadixPrefix)
        glUniform1i(glGetUniformLocation(csRadixPrefix, "uBlocks"), blocks)
        glDispatchCompute(1, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // Scatter
        glUseProgram(csRadixScatter)
        glUniform1i(glGetUniformLocation(csRadixScatter, "uCount"), N)
        glUniform1i(glGetUniformLocation(csRadixScatter, "uPass"), pass)
        glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // ping-pong: morton<->mortonT, indexIn<->indexT
        swapSSBO(1, 3); swapSSBO(2, 4)
    }

    // Перепорядочить тела
    glUseProgram(csReindexBodies)
    glUniform1i(glGetUniformLocation(csReindexBodies, "uCount"), N)
    glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    // Узлы (внутренних N-1)
    glUseProgram(csBuildNodes)
    glUniform1i(glGetUniformLocation(csBuildNodes, "uCount"), N)
    glDispatchCompute(ceilDiv(N - 1, Config.WORKGROUP), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    // Массы: листья
    glUseProgram(csMassLeaves)
    glUniform1i(glGetUniformLocation(csMassLeaves, "uCount"), N)
    glDispatchCompute(ceilDiv(N, Config.WORKGROUP), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    // Массы: подъём уровнями
    glUseProgram(csMassLevels)
    glUniform1i(glGetUniformLocation(csMassLevels, "uCount"), N)
    for (lvl in Config.MAX_DEPTH downTo 0) {
        glUniform1i(glGetUniformLocation(csMassLevels, "uLevel"), lvl)
        glDispatchCompute(ceilDiv(N - 1, Config.WORKGROUP), 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
    }

    // Root = внутренний узел без parent
    glUseProgram(csRootClear)
    glDispatchCompute(1,1,1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    glUseProgram(csFindRoot)
    glUniform1i(glGetUniformLocation(csFindRoot, "uCount"), N)
    glDispatchCompute(ceilDiv(N - 1, Config.WORKGROUP), 1, 1)
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
}

private fun swapSSBO(bindingA: Int, bindingB: Int) {
    val a = IntArray(1); val b = IntArray(1)
    glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, bindingA, a)
    glGetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, bindingB, b)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingA, b[0])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingB, a[0])
}

/* ============================== ШЕЙДЕРЫ ============================== */
object Shaders {

    /* ---- Общая разметка буферов/ubo ---- */
    private const val COMMON = """
#version 460
layout(std430, binding=0)  buffer Bodies  { float bodies[];  }; // pos.xy, vel.xy, mass, pad
layout(std430, binding=1)  buffer Keys    { uint  morton[];  };
layout(std430, binding=2)  buffer Index   { uint  indexIn[]; };
layout(std430, binding=3)  buffer KeysT   { uint  mortonT[]; };
layout(std430, binding=4)  buffer IndexT  { uint  indexT[];  };
layout(std430, binding=5)  buffer Nodes   { uint  nodes[];   };
layout(std430, binding=6)  buffer Stack   { uint  stackBuf[];};
layout(std430, binding=7)  buffer GridH   { uint  gridHead[];};
layout(std430, binding=8)  buffer GridL   { uint  gridList[];};
layout(std430, binding=9)  buffer Dead    { uint  deadFlags[]; };
layout(std430, binding=10) buffer Accel   { float accel[];    }; // ax, ay
layout(std430, binding=11) buffer Counts  { uint  counts[];    }; // [blocks*16]
layout(std430, binding=12) buffer Prefix  { uint  prefix[];    }; // [blocks*16]
layout(std430, binding=13) buffer Globals { uint  globals[];   }; // [16]
layout(std430, binding=14) buffer GScan   { uint  gprefix[];   }; // [16]
layout(std430, binding=15) buffer Root    { uint  rootIdx[];   }; // [1]

layout(std140, binding=0) uniform Sim {
    float uG;
    float uDT;
    float uSoft;
    float uTheta;
    int   uW;
    int   uH;
    float uMergeMinDist;
    float uMergeMaxMass;
};

const int   STRIDE = 6;
const float WIDTH  = float(${Config.WIDTH});
const float HEIGHT = float(${Config.HEIGHT});
const int   STACK_CAP = ${Config.STACK_CAP};

void bodyRead(int i, out vec2 pos, out vec2 vel, out float m) {
    int o = i*STRIDE;
    pos = vec2(bodies[o+0], bodies[o+1]);
    vel = vec2(bodies[o+2], bodies[o+3]);
    m   = bodies[o+4];
}
void bodyWrite(int i, vec2 pos, vec2 vel, float m) {
    int o = i*STRIDE;
    bodies[o+0]=pos.x; bodies[o+1]=pos.y;
    bodies[o+2]=vel.x; bodies[o+3]=vel.y;
    bodies[o+4]=m;
}
void accelWrite(int i, vec2 a){ int o=i*2; accel[o+0]=a.x; accel[o+1]=a.y; }
vec2 accelRead(int i){ int o=i*2; return vec2(accel[o+0], accel[o+1]); }
"""

    /* ---- Узел дерева ----
       [0]: size Q16.16
       [1]: com.x (float bits)
       [2]: com.y
       [3]: mass
       [4..7]: child[0..3] (uint, 0xFFFFFFFF нет; листья: child0 = (leafBodyIdx | 0x80000000))
       [8]: start(body idx, для листа)
       [9]: end  (exclusive)
       [10]: parent (uint или 0xFFFFFFFF)
       [11]: depth (uint)
       [12..15]: reserved
    ------------------------------------------------------------------- */
    private const val NODES = """
const uint IDX_NONE = 0xFFFFFFFFu;
const uint IDX_LEAF_FLAG = 0x80000000u;
const int  NODE_U32 = 16;

int  nodeOff(int n){ return n*NODE_U32; }
bool nodeIsLeaf(int n){ return (nodes[nodeOff(n)+4] & IDX_LEAF_FLAG) != 0u; }

float nodeSize(int n){ return float(int(nodes[nodeOff(n)+0]))/65536.0; }
void  nodeSetSize(int n, float s){ nodes[nodeOff(n)+0]=uint(int(round(s*65536.0))); }
vec2  nodeCom(int n){ return vec2(uintBitsToFloat(nodes[nodeOff(n)+1]), uintBitsToFloat(nodes[nodeOff(n)+2])); }
float nodeMass(int n){ return uintBitsToFloat(nodes[nodeOff(n)+3]); }
void  nodeSetComMass(int n, vec2 c, float m){
    nodes[nodeOff(n)+1]=floatBitsToUint(c.x);
    nodes[nodeOff(n)+2]=floatBitsToUint(c.y);
    nodes[nodeOff(n)+3]=floatBitsToUint(m);
}
void  nodeSetLeaf(int n, int bodyIdx, float s, uint parent, uint depth){
    int o=nodeOff(n);
    nodes[o+0]=uint(int(round(s*65536.0)));
    nodes[o+1]=0u; nodes[o+2]=0u; nodes[o+3]=0u;
    nodes[o+4]=uint(bodyIdx)|IDX_LEAF_FLAG;
    nodes[o+5]=IDX_NONE; nodes[o+6]=IDX_NONE; nodes[o+7]=IDX_NONE;
    nodes[o+8]=uint(bodyIdx); nodes[o+9]=uint(bodyIdx+1);
    nodes[o+10]=parent; nodes[o+11]=depth;
}
void  nodeSetInternal(int n, int c0, int c1, int c2, int c3, float s, uint parent, uint depth){
    int o=nodeOff(n);
    nodes[o+0]=uint(int(round(s*65536.0)));
    nodes[o+4]=(c0<0)?IDX_NONE:uint(c0);
    nodes[o+5]=(c1<0)?IDX_NONE:uint(c1);
    nodes[o+6]=(c2<0)?IDX_NONE:uint(c2);
    nodes[o+7]=(c3<0)?IDX_NONE:uint(c3);
    nodes[o+8]=0u; nodes[o+9]=0u;
    nodes[o+10]=parent; nodes[o+11]=depth;
}
"""

    /* ======================= INIT ======================= */
    val initBodies = """
$COMMON
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;

void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount) return;
    float t = float(i)/float(max(uCount-1,1));
    float ang = t*6.2831853*4.0;
    float r = 0.2 + 0.75*t;
    vec2 c = vec2(WIDTH*0.5, HEIGHT*0.5);
    vec2 pos = c + vec2(cos(ang), sin(ang))*r*min(WIDTH,HEIGHT)*0.45;
    vec2 vel = vec2(-sin(ang), cos(ang))*35.0;
    bodyWrite(i,pos,vel,1.0);
    deadFlags[i]=0u;
}
"""

    /* ======================= MORTON ======================= */
    val morton = """
$COMMON
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;

uint expandBits(uint v){
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
uint morton2D(uint x, uint y){ return (expandBits(x) << 1) | expandBits(y); }

void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount) return;
    vec2 p; vec2 v; float m; bodyRead(i,p,v,m);
    float nx=clamp(p.x/WIDTH,0.0,0.999999);
    float ny=clamp(p.y/HEIGHT,0.0,0.999999);
    uint gx=uint(nx*float(${Config.GRID_W}));
    uint gy=uint(ny*float(${Config.GRID_H}));
    morton[i]  = morton2D(gx,gy);
    indexIn[i] = uint(i);
}
"""

    /* ======================= RADIX: Count → Prefix → Scatter ======================= */

    val radixCount = """
#version 460
layout(local_size_x=${Config.WORKGROUP}) in;
layout(std430, binding=1)  buffer Keys   { uint morton[]; };
layout(std430, binding=11) buffer Counts { uint counts[]; }; // [blocks*16]
uniform int uCount;
uniform int uPass;

shared uint sHist[16];

void main(){
    uint gid = gl_GlobalInvocationID.x;
    uint lid = gl_LocalInvocationID.x;
    uint grp = gl_WorkGroupID.x;

    if (lid < 16u) sHist[lid]=0u;
    barrier();

    uint numGroups = gl_NumWorkGroups.x;
    uint L = gl_WorkGroupSize.x;
    for (uint i = gid; i < uint(uCount); i += numGroups*L) {
        uint key = morton[i];
        uint d   = (key >> (uPass*4)) & 0xFu;
        atomicAdd(sHist[d], 1u);
    }
    barrier();

    if (lid < 16u) {
        counts[grp*16u + lid] = sHist[lid];
    }
}
"""

    val radixPrefix = """
#version 460
layout(local_size_x=16) in;
layout(std430, binding=11) buffer Counts  { uint counts[];  };
layout(std430, binding=12) buffer Prefix  { uint prefix[];  };
layout(std430, binding=13) buffer Globals { uint globals[]; };
layout(std430, binding=14) buffer GScan   { uint gprefix[]; };
uniform int uBlocks;

void main(){
    uint d = gl_LocalInvocationID.x; // 0..15
    if (uBlocks<=0) {
        if (d==0u){ for(int k=0;k<16;k++){ globals[k]=0u; gprefix[k]=0u; } }
        return;
    }

    // 1) По каждому digit: префикс по блокам
    uint acc = 0u;
    for (int b=0; b<uBlocks; ++b){
        uint c = counts[uint(b)*16u + d];
        prefix[uint(b)*16u + d] = acc;
        acc += c;
    }
    globals[d] = acc; // totals per digit

    barrier();

    // 2) Эксклюзивный префикс глобальных totals
    if (d==0u) {
        uint run=0u;
        for (int k=0;k<16;k++){ uint v = globals[k]; gprefix[k]=run; run+=v; }
    }
}
"""

    val radixScatter = """
#version 460
layout(local_size_x=${Config.WORKGROUP}) in;
layout(std430, binding=1)  buffer Keys   { uint morton[];  };
layout(std430, binding=2)  buffer Index  { uint indexIn[]; };
layout(std430, binding=3)  buffer KeysT  { uint mortonT[]; };
layout(std430, binding=4)  buffer IndexT { uint indexT[];  };
layout(std430, binding=11) buffer Counts { uint counts[];  };
layout(std430, binding=12) buffer Prefix { uint prefix[];  };
layout(std430, binding=14) buffer GScan  { uint gprefix[]; };
uniform int uCount;
uniform int uPass;

shared uint sOff[16];

void main(){
    uint lid = gl_LocalInvocationID.x;
    uint grp = gl_WorkGroupID.x;
    uint L   = gl_WorkGroupSize.x;
    uint numGroups = gl_NumWorkGroups.x;

    if (lid < 16u) sOff[lid]=0u;
    barrier();

    for (uint i = grp*L + lid; i < uint(uCount); i += numGroups*L) {
        uint key = morton[i];
        uint idx = indexIn[i];
        uint d   = (key >> (uPass*4)) & 0xFu;

        uint base = gprefix[d] + prefix[grp*16u + d];
        uint off  = atomicAdd(sOff[d], 1u);
        uint dst  = base + off;

        mortonT[dst] = key;
        indexT[dst]  = idx;
    }
}
"""

    /* ======================= REINDEX BODIES ======================= */
    val reindexBodies = """
$COMMON
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;
void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount) return;
    uint src=indexIn[i];
    vec2 p; vec2 v; float m; bodyRead(int(src),p,v,m);
    bodyWrite(i,p,v,m);
    deadFlags[i]=deadFlags[int(src)];
}
"""

    /* ======================= BUILD NODES (LBVH) ======================= */
    val buildNodesLBVH = """
$COMMON
$NODES
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;

int cplInt(uint a, uint b){ if(a==b) return 32; return 31 - findMSB(a ^ b); }
int cplMortonIdx(int i, int j){
    if(j<0 || j>=uCount) return -1;
    uint a=morton[i], b=morton[j];
    int c=cplInt(a,b);
    if(a==b) { // различаем индексы, чтобы не схлопывалось
        int di = (i==j)?32:(31 - findMSB(uint(i ^ j)));
        return 32 + di;
    }
    return c;
}
int depthFromCpl(int cplBits){ return max(0, cplBits/2); }
float sizeFromDepth(int depth){
    float root=float(max(uW,uH));
    return root / pow(2.0, float(depth));
}

void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount-1) return;

    // Направление диапазона
    int d  = (cplMortonIdx(i,i+1)-cplMortonIdx(i,i-1))>=0 ? 1 : -1;
    int cplm = cplMortonIdx(i, i-d);
    int lmax = 2;
    while (cplMortonIdx(i, i + lmax*d) > cplm) lmax <<= 1;

    int l=0;
    for(int t=lmax>>1; t>=1; t>>=1){
        if (cplMortonIdx(i, i+(l+t)*d) > cplm) l += t;
    }
    int j = i + l*d;
    int L = min(i,j), R = max(i,j);
    int cplNode = cplMortonIdx(L,R);
    int depth   = depthFromCpl(cplNode);
    float size  = sizeFromDepth(depth);

    // Найти split
    int c = cplMortonIdx(i,j);
    int s=0; int step=l;
    do{
        step=(step+1)>>1;
        if (cplMortonIdx(i, i+(s+step)*d) > c) s+=step;
    }while(step>1);
    int split = i + s*d;

    // Дети
    int leftIsLeaf  = (min(i,j)==split)?1:0;
    int rightIsLeaf = (split+1==max(i,j))?1:0;

    int internal = i;
    uint parent = IDX_NONE;

    // Левый ребёнок
    int leftIdx;
    if (leftIsLeaf==1){
        leftIdx = uCount-1 + split;
        nodeSetLeaf(leftIdx, split, size, uint(internal), uint(depth+1));
    } else {
        leftIdx = split;
        nodes[nodeOff(leftIdx)+10] = uint(internal);   // проставляем parent
    }
    nodeSetInternal(internal, leftIdx, -1, -1, -1, size, parent, uint(depth));

    // Правый ребёнок
    int rightIdx;
    if (rightIsLeaf==1){
        rightIdx = uCount-1 + (split+1);
        nodeSetLeaf(rightIdx, split+1, size, uint(internal), uint(depth+1));
        nodes[nodeOff(internal)+5] = uint(rightIdx);
    } else {
        nodes[nodeOff(internal)+5] = uint(split+1);
        rightIdx = split+1;
        nodes[nodeOff(rightIdx)+10] = uint(internal);  // проставляем parent
    }

    nodes[nodeOff(internal)+6] = IDX_NONE;
    nodes[nodeOff(internal)+7] = IDX_NONE;
}
"""

    /* ======================= MASS BOTTOM-UP ======================= */
    val massLeaves = """
$COMMON
$NODES
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;

void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount) return;
    int leaf = uCount-1 + i;
    vec2 p; vec2 v; float m; bodyRead(i,p,v,m);
    nodeSetComMass(leaf, p, m);
}
"""

    val massReduceLevels = """
$COMMON
$NODES
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;
uniform int uLevel;

void main(){
    int n=int(gl_GlobalInvocationID.x);
    if(n >= (uCount-1)) return; // только внутренние
    if (nodeIsLeaf(n)) return;

    float s = nodeSize(n);
    float root = float(max(uW,uH));
    if (s <= 0.0) return;
    int depth = int(round(log2(root/s)));
    if (depth != uLevel) return;

    int o = nodeOff(n);
    uint c0=nodes[o+4], c1=nodes[o+5], c2=nodes[o+6], c3=nodes[o+7];

    float msum=0.0;
    vec2  csum=vec2(0.0);
    for(int k=0;k<4;k++){
        uint ci = (k==0?c0:(k==1?c1:(k==2?c2:c3)));
        if (ci==IDX_NONE) continue;
        int child = int(ci & 0x7FFFFFFFu);
        float m  = nodeMass(child);
        if (m>0.0){ vec2 c = nodeCom(child); msum += m; csum += c*m; }
    }
    if (msum>0.0) nodeSetComMass(n, csum/msum, msum);
    else          nodeSetComMass(n, vec2(0.0), 0.0);
}
"""

    /* ======================= ROOT (clear + find) ======================= */
    val rootClear = """
#version 460
layout(local_size_x=1) in;
layout(std430, binding=15) buffer Root  { uint rootIdx[]; };
void main(){ rootIdx[0] = 0xFFFFFFFFu; }
"""


    val findRoot = """
#version 460
layout(local_size_x=${Config.WORKGROUP}) in;
layout(std430, binding=5)  buffer Nodes { uint nodes[]; };
layout(std430, binding=15) buffer Root  { uint rootIdx[]; };

const uint IDX_NONE = 0xFFFFFFFFu;
const int  NODE_U32 = 16;
int nodeOff(int n){ return n*NODE_U32; }

uniform int uCount;

void main(){
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uCount-1)) return; // только внутренние [0..N-2]
    uint parent = nodes[nodeOff(int(i))+10];
    if (parent == IDX_NONE) {
        atomicMin(rootIdx[0], i);
    }
}
"""

    /* ======================= ACCEL / KICK / DRIFT ======================= */
    val accumulateForces = """
$COMMON
$NODES
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;

void safePush(int base, inout int sp, uint v){
    if (sp < STACK_CAP) { stackBuf[base + sp] = v; sp++; }
}

void addPoint(vec2 p, float m2, inout vec2 f, vec2 bpos, float bm){
    vec2 d = p - bpos;
    float r2 = dot(d,d) + uSoft*uSoft;
    float invR = inversesqrt(r2);
    float invR2= 1.0/r2;
    float F = uG * bm * m2 * invR2;
    f += F * d * invR;
}

void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount){ return; }
    if (deadFlags[i]!=0u) { accelWrite(i, vec2(0)); return; }

    vec2 pos; vec2 vel; float m; bodyRead(i,pos,vel,m);

    uint root = rootIdx[0];
    if (root==0xFFFFFFFFu) { accelWrite(i, vec2(0)); return; }

    int base = i*STACK_CAP; int sp=0;
    safePush(base, sp, root);

    vec2 F=vec2(0.0);
    while (sp>0){
        int n = int(stackBuf[base + (--sp)]);
        if (n<0) continue;
        if (nodeIsLeaf(n)){
            uint raw = nodes[nodeOff(n)+4];
            int bi = int(raw & 0x7FFFFFFFu);
            if (bi!=i && deadFlags[bi]==0u){
                vec2 p; vec2 v; float mm; bodyRead(bi,p,v,mm);
                addPoint(p, mm, F, pos, m);
            }
        } else {
            float s = nodeSize(n);
            vec2  c = nodeCom(n);
            float mm= nodeMass(n);
            vec2 d = c - pos;
            float r2 = dot(d,d) + uSoft*uSoft;
            float s2 = s*s;
            if (s2 < (uTheta*uTheta)*r2){
                addPoint(c, mm, F, pos, m);
            } else {
                int o=nodeOff(n);
                uint c0=nodes[o+4], c1=nodes[o+5], c2=nodes[o+6], c3=nodes[o+7];
                if (c0!=0xFFFFFFFFu) safePush(base, sp, c0);
                if (c1!=0xFFFFFFFFu) safePush(base, sp, c1);
                if (c2!=0xFFFFFFFFu) safePush(base, sp, c2);
                if (c3!=0xFFFFFFFFu) safePush(base, sp, c3);
            }
        }
    }
    vec2 a = F / max(m,1e-6);
    accelWrite(i,a);
}
"""

    val kickHalf = """
$COMMON
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;
void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount) return;
    if(deadFlags[i]!=0u) return;
    vec2 p; vec2 v; float m; bodyRead(i,p,v,m);
    vec2 a = accelRead(i);
    v += a * (0.5 * uDT);
    bodyWrite(i,p,v,m);
}
"""

    val drift = """
$COMMON
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;
void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount) return;
    if(deadFlags[i]!=0u) return;
    vec2 p; vec2 v; float m; bodyRead(i,p,v,m);
    p += v * uDT;
    if(p.x<0.0) p.x+=WIDTH; if(p.x>=WIDTH) p.x-=WIDTH;
    if(p.y<0.0) p.y+=HEIGHT; if(p.y>=HEIGHT) p.y-=HEIGHT;
    bodyWrite(i,p,v,m);
}
"""

    /* ======================= MERGE ======================= */
    val gridClear = """
#version 460
layout(local_size_x=${Config.WORKGROUP}) in;
layout(std430, binding=7)  buffer GridH { uint gridHead[]; };
layout(std430, binding=8)  buffer GridL { uint gridList[]; };
void main(){
    uint i = gl_GlobalInvocationID.x;
    if (i < ${Config.GRID_CELLS}u) gridHead[i] = 0xFFFFFFFFu;
    uint total = ${Config.GRID_CELLS * Config.GRID_BUCKET}u;
    if (i < total) gridList[i] = 0xFFFFFFFFu;
}
"""
    val gridFill = """
$COMMON
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;
uint hashCell(ivec2 c){ uint h=uint(c.x)*73856093u ^ uint(c.y)*19349663u; return h & uint(${Config.GRID_CELLS-1}); }
void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount) return;
    if(deadFlags[i]!=0u) return;
    vec2 p; vec2 v; float m; bodyRead(i,p,v,m);
    ivec2 cell = ivec2(int(p.x)>>5, int(p.y)>>5);
    uint h = hashCell(cell);
    for (int k=0;k<${Config.GRID_BUCKET};k++){
        uint idx = atomicCompSwap(gridList[h*${Config.GRID_BUCKET}+k], 0xFFFFFFFFu, uint(i));
        if (idx==0xFFFFFFFFu) { gridHead[h]=h; break; }
    }
}
"""
    val mergeBodies = """
$COMMON
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;
uint hashCell(ivec2 c){ uint h=uint(c.x)*73856093u ^ uint(c.y)*19349663u; return h & uint(${Config.GRID_CELLS-1}); }
void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount) return;
    if(deadFlags[i]!=0u) return;

    vec2 pi; vec2 vi; float mi; bodyRead(i,pi,vi,mi);
    if (mi <= uMergeMaxMass) return;

    float r2 = uMergeMinDist*uMergeMinDist;
    ivec2 cell = ivec2(int(pi.x)>>5, int(pi.y)>>5);

    for(int oy=-1;oy<=1;++oy) for(int ox=-1;ox<=1;++ox){
        ivec2 cc = cell + ivec2(ox,oy);
        uint h = hashCell(cc);
        for (int k=0;k<${Config.GRID_BUCKET};k++){
            uint j = gridList[h*${Config.GRID_BUCKET}+k];
            if (j==0xFFFFFFFFu || j==uint(i)) continue;
            if (deadFlags[int(j)]!=0u) continue;
            vec2 pj; vec2 vj; float mj; bodyRead(int(j),pj,vj,mj);
            vec2 d=pj-pi;
            if (dot(d,d) < r2){
                deadFlags[int(j)]=1u;
                mi += mj;
            }
        }
    }
    bodyWrite(i,pi,vi,mi);
}
"""
    val compactDead = """
$COMMON
layout(local_size_x=${Config.WORKGROUP}) in;
uniform int uCount;
void main(){
    int i=int(gl_GlobalInvocationID.x);
    if(i>=uCount) return;
    if(deadFlags[i]!=0u){
        vec2 p; vec2 v; float m; bodyRead(i,p,v,m);
        bodyWrite(i, vec2(-1e6,-1e6), v, 0.0);
    }
}
"""

    /* ======================= RENDER (SSBO) ======================= */
    val vsRender = """
#version 460
layout(std430, binding=0) buffer Bodies { float bodies[]; };
uniform vec2 uScreen;
const int STRIDE=6;
void main(){
    int i = gl_VertexID;
    int o = i*STRIDE;
    vec2 pos = vec2(bodies[o+0], bodies[o+1]);
    gl_Position = vec4((pos / uScreen * 2.0 - 1.0) * vec2(1,-1), 0.0, 1.0);
    gl_PointSize = 1.5;
}
"""
    val fsRender = """
#version 460
out vec4 FragColor;
void main(){ FragColor = vec4(1,1,1,1); }
"""
}
