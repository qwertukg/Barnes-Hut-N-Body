@file:JvmName("BarnesHutGPU")
package gpu

import org.lwjgl.glfw.GLFW.*
import org.lwjgl.glfw.GLFWErrorCallback
import org.lwjgl.opengl.GL
import org.lwjgl.opengl.GL46C.*
import org.lwjgl.system.MemoryUtil.NULL
import kotlin.math.max
import kotlin.random.Random

private object Cfg {
    const val WIDTH = 1280
    const val HEIGHT = 720
    const val LOCAL_SIZE = 256
    const val MAX_BODIES = 100_000
    const val MAX_NODES  = 8 * MAX_BODIES      // запас, чтобы не упираться
    const val START_BODIES = 20_000

    // Физика (как у тебя)
    const val G = 80.0
    const val DT = 0.005
    const val SOFT = 1.0
    const val THETA = 0.6
    const val MERGE_MAX_MASS = 4_000.0
    const val MERGE_MIN_DIST = 2.0  // 0 — отключить merge

    // Размеры структур (std430)
    const val BODY_BYTES = 48L      // dvec2 + dvec2 + double + int + int
    const val NODE_BYTES = 128L     // с запасом (выравнивание std430 у double/dvec2)
}

/* ================= GLFW / GL utils ================= */

private fun createWindow(): Long {
    GLFWErrorCallback.createPrint(System.err).set()
    if (!glfwInit()) error("GLFW init failed")
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE)
    val win = glfwCreateWindow(Cfg.WIDTH, Cfg.HEIGHT, "Barnes–Hut GPU (GL46, SSBO)", NULL, NULL)
    if (win == NULL) error("Window create failed")
    glfwMakeContextCurrent(win)
    glfwSwapInterval(0)
    GL.createCapabilities()
    return win
}

private fun compile(type: Int, src: String): Int {
    val sh = glCreateShader(type)
    glShaderSource(sh, src)
    glCompileShader(sh)
    if (glGetShaderi(sh, GL_COMPILE_STATUS) == GL_FALSE) {
        val log = glGetShaderInfoLog(sh)
        throw IllegalStateException("Shader compile error:\n$log")
    }
    return sh
}

private fun linkProgram(shaders: IntArray): Int {
    val p = glCreateProgram()
    for (s in shaders) glAttachShader(p, s)
    glLinkProgram(p)
    if (glGetProgrami(p, GL_LINK_STATUS) == GL_FALSE) {
        val log = glGetProgramInfoLog(p)
        throw IllegalStateException("Program link error:\n$log")
    }
    for (s in shaders) { glDetachShader(p, s); glDeleteShader(s) }
    return p
}

private fun ssbo(binding: Int, bytes: Long): Int {
    val id = glGenBuffers()
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, id)
    glBufferData(GL_SHADER_STORAGE_BUFFER, bytes, GL_DYNAMIC_DRAW)
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, id)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return id
}

private fun vaoForDraw(count: Int): Int {
    val vao = glGenVertexArrays()
    val vbo = glGenBuffers()
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, (count * 4L), GL_STATIC_DRAW) // uint per vertex
    val arr = java.nio.ByteBuffer
        .allocateDirect(count * 4)
        .order(java.nio.ByteOrder.nativeOrder())
        .asIntBuffer()
    for (i in 0 until count) arr.put(i)
    arr.flip()
    glBufferSubData(GL_ARRAY_BUFFER, 0, arr)
    glEnableVertexAttribArray(0)
    glVertexAttribIPointer(0, 1, GL_UNSIGNED_INT, 4, 0L)
    glBindVertexArray(0)
    return vao
}

private fun groups(n: Int) = (n + Cfg.LOCAL_SIZE - 1) / Cfg.LOCAL_SIZE

/* ================== Init data ================== */

private fun initBodies(permSSBO: Int, bodySSBO: Int, bodyCount: Int) {
    // perm[i] = i
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, permSSBO)
    val pbuf = glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, (Cfg.MAX_BODIES * 4L),
        GL_MAP_WRITE_BIT or GL_MAP_INVALIDATE_BUFFER_BIT
    )!!.order(java.nio.ByteOrder.nativeOrder()).asIntBuffer()
    for (i in 0 until bodyCount) pbuf.put(i)
    for (i in bodyCount until Cfg.MAX_BODIES) pbuf.put(0)
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

    // ---- Body в std430 = 48 байт ----
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bodySSBO)
    val total = Cfg.MAX_BODIES * Cfg.BODY_BYTES
    val bb = glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, total,
        GL_MAP_WRITE_BIT or GL_MAP_INVALIDATE_BUFFER_BIT
    )!!.order(java.nio.ByteOrder.nativeOrder())

    fun putBody(i: Int, x: Double, y: Double, vx: Double, vy: Double, m: Double) {
        val base = i * Cfg.BODY_BYTES.toInt()
        bb.putDouble(base +  0, x)   // pos.x
        bb.putDouble(base +  8, y)   // pos.y
        bb.putDouble(base + 16, vx)  // vel.x
        bb.putDouble(base + 24, vy)  // vel.y
        bb.putDouble(base + 32, m)   // m
        bb.putInt   (base + 40, 1)   // alive
        bb.putInt   (base + 44, 0)   // pad
    }

    val rnd = Random(1)
    val cx = Cfg.WIDTH * 0.5
    val cy = Cfg.HEIGHT * 0.5
    val rMax = (max(Cfg.WIDTH, Cfg.HEIGHT) * 0.45)
    for (i in 0 until bodyCount) {
        val a = rnd.nextDouble() * Math.PI * 2.0
        val r = Math.sqrt(rnd.nextDouble()) * rMax
        val x = cx + r * kotlin.math.cos(a)
        val y = cy + r * kotlin.math.sin(a)
        val v = 30.0 + rnd.nextDouble() * 30.0
        val vx = -v * kotlin.math.sin(a)
        val vy =  v * kotlin.math.cos(a)
        val m  = 1.0 + rnd.nextDouble() * 2.0
        putBody(i, x, y, vx, vy, m)
    }
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
}

/* ================ GLSL sources (fixed) ================ */

private val commonSSBO = """
// НЕТ #version здесь!

struct Body {
    dvec2 pos;
    dvec2 vel;
    double m;
    int alive;
    int pad;
};

struct Node {
    dvec2 com;
    double mass;
    dvec2 center;
    double h;
    int child[4];
    int body;
    int level;
    int lock;
    int isLeaf;
    int pad0;
};

layout(std430, binding = 0) buffer Bodies { int perm[]; };
layout(std430, binding = 1) buffer BodyData { Body bodies[]; };
layout(std430, binding = 2) buffer Nodes    { Node nodes[]; };

layout(std430, binding = 3) buffer Globals {
    int  nodeCount;
    int  bodyCount;
    int  maxLevel;
    double dt;
    double theta;
    double soft2;
    double G;
    double mergeMaxMass;
    double mergeMinDist;
    int  rootIndex;
    int  needRebuild;
};
""".trimIndent()

private fun inject(src: String) = src.replace("#include_common", commonSSBO)

private val csReset = """
#version 460 core
layout(local_size_x = 256) in;
#include_common
uniform dvec2 uWindow;
void main(){
    if (gl_GlobalInvocationID.x != 0) return;

    // НЕ трогаем bodyCount — он уже установлен с CPU один раз
    nodeCount = 1;
    rootIndex = 0;
    maxLevel  = 0;
    needRebuild = 0;

    dt = ${Cfg.DT};
    theta = ${Cfg.THETA};
    soft2 = ${Cfg.SOFT*Cfg.SOFT};
    G = ${Cfg.G};
    mergeMaxMass = ${Cfg.MERGE_MAX_MASS};
    mergeMinDist = ${Cfg.MERGE_MIN_DIST};

    Node n;
    n.com    = dvec2(0.0);
    n.mass   = 0.0;
    n.center = dvec2(uWindow.x*0.5, uWindow.y*0.5);
    double _half = (uWindow.x > uWindow.y ? uWindow.x : uWindow.y)*0.5 + 2.0;
    n.h = _half*0.5;
    n.child[0]=n.child[1]=n.child[2]=n.child[3]=-1;
    n.body  = -1;
    n.level = 0;
    n.lock  = 0;
    n.isLeaf= 1;
    n.pad0  = 0;
    nodes[0]= n;
}
""".trimIndent()

private val csBuild = """
#version 460 core
layout(local_size_x = ${Cfg.LOCAL_SIZE}) in;
#include_common

uvec2 dpack(double x){ return unpackDouble2x32(x); }

int newNode(int parentLevel, dvec2 center, double h){
    int idx = atomicAdd(nodeCount, 1);
    // Безопасность: если вдруг переполнили пул — откат на последний валидный
    if (idx < 0) idx = 0;
    Node n;
    n.com = dvec2(0.0);
    n.mass = 0.0;
    n.center = center;
    n.h = h;
    n.child[0]=n.child[1]=n.child[2]=n.child[3]=-1;
    n.body = -1;
    n.level = parentLevel + 1;
    n.lock = 0;
    n.isLeaf = 1;
    n.pad0 = 0;
    nodes[idx] = n;
    atomicMax(maxLevel, n.level);
    return idx;
}

int childIndex(Node n, dvec2 p){
    int ix = (p.x < n.center.x) ? 0 : 1;
    int iy = (p.y < n.center.y) ? 0 : 2;
    return ix + iy;
}

void insertBody(int bi){
    Body b = bodies[bi];
    if (b.alive == 0) return;

    int cur = rootIndex;

    for (int safety=0; safety<1024; ++safety){
        // lock
        while (atomicCompSwap(nodes[cur].lock, 0, 1) != 0) { }
        Node n = nodes[cur];

        if (n.isLeaf == 1){
            if (n.body < 0){
                n.body = bi;
                nodes[cur] = n;
                nodes[cur].lock = 0;
                return;
            } else {
                int oldBody = n.body;
                n.body = -1;
                n.isLeaf = 0;
                double hh = n.h * 0.5;
                int c0 = newNode(n.level, dvec2(n.center.x - hh, n.center.y - hh), hh);
                int c1 = newNode(n.level, dvec2(n.center.x + hh, n.center.y - hh), hh);
                int c2 = newNode(n.level, dvec2(n.center.x - hh, n.center.y + hh), hh);
                int c3 = newNode(n.level, dvec2(n.center.x + hh, n.center.y + hh), hh);
                n.child[0]=c0; n.child[1]=c1; n.child[2]=c2; n.child[3]=c3;
                nodes[cur] = n;
                nodes[cur].lock = 0;

                if (n.h < 1e-3){
                    Body ob = bodies[oldBody];
                    double eps = 1e-3;
                    uvec2 wx = dpack(ob.pos.x);
                    uvec2 wy = dpack(ob.pos.y);
                    ob.pos.x += ((wx.x & 1u)==0u) ? +eps : -eps;
                    ob.pos.y += ((wy.x & 1u)==0u) ? -eps : +eps;
                    bodies[oldBody] = ob;
                }

                // вставить oldBody вниз
                int target = cur;
                Body ob2 = bodies[oldBody];
                for (int s2=0; s2<1024; ++s2){
                    while (atomicCompSwap(nodes[target].lock,0,1)!=0) {}
                    Node nn = nodes[target];
                    if (nn.isLeaf==1 && nn.body<0){
                        nn.body = oldBody; nodes[target]=nn; nodes[target].lock=0; break;
                    }
                    int ci = childIndex(nn, ob2.pos);
                    int nxt = nn.child[ci];
                    nodes[target].lock=0;
                    if (nxt < 0) { // страховка
                        break;
                    }
                    target = nxt;
                }
                // продолжаем вставку bi с вершины cur — цикл сам вернёт нас
            }
        } else {
            int ci = childIndex(n, b.pos);
            int nxt = n.child[ci];
            nodes[cur].lock = 0;
            if (nxt < 0) { // страховка от гонки
                continue;
            }
            cur = nxt;
        }
    }
}

void main(){
    uint g = gl_GlobalInvocationID.x;
    if (g >= uint(bodyCount)) return;
    int bi = perm[g];
    insertBody(bi);
}
""".trimIndent()

private val csMass = """
#version 460 core
layout(local_size_x = ${Cfg.LOCAL_SIZE}) in;
#include_common
uniform int uLevel;
void main(){
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(nodeCount)) return;
    Node n = nodes[idx];
    if (n.level != uLevel) return;

    if (n.isLeaf == 1){
        if (n.body >= 0){
            Body b = bodies[n.body];
            if (b.alive == 1){ n.mass = b.m; n.com = b.pos; }
            else { n.mass = 0.0; n.com = n.center; }
        } else { n.mass = 0.0; n.com = n.center; }
        nodes[idx] = n;
        return;
    }

    double mSum = 0.0;
    dvec2 c = dvec2(0.0);
    for (int k=0;k<4;k++){
        int ci = n.child[k]; if (ci<0) continue;
        Node ch = nodes[ci];
        if (ch.mass > 0.0){ mSum += ch.mass; c += ch.com * ch.mass; }
    }
    n.mass = mSum;
    n.com  = (mSum>0.0)?(c/mSum):n.center;
    nodes[idx]=n;
}
""".trimIndent()

private val csAccel = """
#version 460 core
layout(local_size_x = ${Cfg.LOCAL_SIZE}) in;
#include_common
layout(std430, binding = 5) buffer AccelX { double ax[]; };
layout(std430, binding = 6) buffer AccelY { double ay[]; };

void pointForce(in dvec2 p, in double m, inout double fx, inout double fy,
                in dvec2 bpos, in double bm, in double soft2, in double G){
    dvec2 d = p - bpos;
    double r2 = dot(d,d) + soft2;
    double invR = inversesqrt(r2);
    double invR2 = 1.0 / r2;
    double f = G * bm * m * invR2;
    double c = f * invR;
    fx += c * d.x;
    fy += c * d.y;
}

void main(){
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(bodyCount)) return;
    Body b = bodies[gid];
    if (b.alive == 0){ ax[gid]=0.0; ay[gid]=0.0; return; }

    double theta2 = theta*theta;
    int stack[64]; int sp=0; stack[sp++]=rootIndex;
    double fx=0.0, fy=0.0;

    while (sp>0){
        int ni = stack[--sp];
        Node n = nodes[ni];
        if (n.mass == 0.0) continue;

        if (n.isLeaf == 1){
            if (n.body >= 0 && n.body != int(gid)){
                Body sb = bodies[n.body];
                if (sb.alive==1) pointForce(sb.pos, sb.m, fx,fy, b.pos, b.m, soft2, G);
            }
            continue;
        }
        dvec2 d = n.com - b.pos;
        double dist2 = dot(d,d) + soft2;
        double s2 = (n.h*2.0); s2 = s2*s2;

        if (s2 < theta2 * dist2){
            pointForce(n.com, n.mass, fx,fy, b.pos, b.m, soft2, G);
        } else {
            for (int k=0;k<4;k++){ int ci=n.child[k]; if (ci>=0) stack[sp++]=ci; }
        }
    }
    ax[gid] = fx / b.m;
    ay[gid] = fy / b.m;
}
""".trimIndent()

private val csKick = """
#version 460 core
layout(local_size_x = ${Cfg.LOCAL_SIZE}) in;
#include_common
layout(std430, binding = 5) buffer AccelX { double ax[]; };
layout(std430, binding = 6) buffer AccelY { double ay[]; };
void main(){
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(bodyCount)) return;
    if (bodies[gid].alive == 0) return;
    double dtHalf = dt*0.5;
    bodies[gid].vel.x += ax[gid]*dtHalf;
    bodies[gid].vel.y += ay[gid]*dtHalf;
}
""".trimIndent()

private val csDrift = """
#version 460 core
layout(local_size_x = ${Cfg.LOCAL_SIZE}) in;
#include_common
void main(){
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= uint(bodyCount)) return;
    if (bodies[gid].alive == 0) return;
    bodies[gid].pos += bodies[gid].vel * dt;
}
""".trimIndent()

private val csMerge = """
#version 460 core
layout(local_size_x = ${Cfg.LOCAL_SIZE}) in;
#include_common
layout(std430, binding = 7) buffer Victims  { int victimFlags[]; };
layout(std430, binding = 8) buffer BodyLocks{ int bodyLock[]; };

void main(){
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(bodyCount)) return;
    Body bi = bodies[i];
    if (bi.alive == 0) return;
    if (mergeMinDist <= 0.0) return;

    double minD2 = mergeMinDist*mergeMinDist;

    if (bi.m > mergeMaxMass){
        for (uint j=0; j<uint(bodyCount); ++j){
            if (j==i) continue;
            Body bj = bodies[j];
            if (bj.alive == 0) continue;
            dvec2 d = bj.pos - bi.pos;
            if (dot(d,d) < minD2){
                if (atomicCompSwap(victimFlags[j], 0, 1) == 0){
                    while (atomicCompSwap(bodyLock[i], 0, 1) != 0) {}
                    double mi = bodies[i].m;
                    mi += bj.m;
                    bodies[i].m = mi;
                    bodyLock[i] = 0;
                }
            }
        }
    }
}
""".trimIndent()

private val csCompact = """
#version 460 core
layout(local_size_x = ${Cfg.LOCAL_SIZE}) in;
#include_common
layout(std430, binding = 7)  buffer Victims  { int victimFlags[]; };
layout(std430, binding = 9)  buffer ScanTemp { int tmp[]; };
layout(std430, binding = 10) buffer NewCount { int outCount[]; };
shared uint blockAlive[${Cfg.LOCAL_SIZE}];

void main(){
    uint gid = gl_GlobalInvocationID.x;
    if (gid < uint(bodyCount)){
        if (victimFlags[gid] == 1 && bodies[gid].alive == 1) bodies[gid].alive = 0;
    }

    uint lane = gl_LocalInvocationID.x;
    uint alive = (gid < uint(bodyCount) && bodies[gid].alive==1) ? 1u : 0u;
    blockAlive[lane] = alive;
    memoryBarrierShared(); barrier();

    for (uint off=1u; off<gl_WorkGroupSize.x; off<<=1u){
        uint t = (lane>=off) ? blockAlive[lane-off] : 0u;
        barrier();
        blockAlive[lane] += t;
        barrier();
    }
    uint localIndex = (alive==1u) ? (blockAlive[lane]-1u) : 0xffffffffu;

    if (lane == gl_WorkGroupSize.x-1u){
        tmp[gl_WorkGroupID.x] = int(blockAlive[lane]);
    }
    barrier();

    if (gl_WorkGroupID.x==0u && lane==0u){
        int sum=0;
        for (uint g=0u; g<gl_NumWorkGroups.x; ++g){
            int s = tmp[g];
            tmp[g]=sum;
            sum+=s;
        }
        outCount[0]=sum;
    }
    barrier();

    if (alive==1u){
        int globalIndex = tmp[gl_WorkGroupID.x] + int(localIndex);
        perm[globalIndex] = int(gid);
    }

    if (gl_GlobalInvocationID.x==0u){
        bodyCount = outCount[0];
        needRebuild = 1;
    }
}
""".trimIndent()

private val vsPoints = """
#version 460 core
layout(location=0) in uint inIndex;
#include_common
uniform vec2 uViewport;
out float vMass;
void main(){
    int bi = perm[int(inIndex)];
    Body b = bodies[bi];
    float x = float((b.pos.x / uViewport.x) * 2.0 - 1.0);
    float y = float((b.pos.y / uViewport.y) * 2.0 - 1.0);
    gl_Position = vec4(x,y,0.0,1.0);
    vMass = float(b.m);
    gl_PointSize = max(1.0, sqrt(vMass));
}
""".trimIndent()

private val fsPoints = """
#version 460 core
in float vMass;
out vec4 fragColor;
void main(){
    float a = clamp(log2(1.0 + max(vMass,1.0))*0.10, 0.15, 1.0);
    fragColor = vec4(1.0,1.0,1.0,a);
}
""".trimIndent()

/* ============================ MAIN ============================ */

fun main() {
    val win = createWindow()

    // --- SSBOs ---
    val ssboPerm     = ssbo(0, Cfg.MAX_BODIES * 4L)
    val ssboBodies   = ssbo(1, Cfg.MAX_BODIES * Cfg.BODY_BYTES)
    val ssboNodes    = ssbo(2, Cfg.MAX_NODES  * Cfg.NODE_BYTES)
    val ssboGlobals  = ssbo(3, 256L)          // достаточно
    val ssboAx       = ssbo(5, Cfg.MAX_BODIES * 8L)
    val ssboAy       = ssbo(6, Cfg.MAX_BODIES * 8L)
    val ssboVictims  = ssbo(7, Cfg.MAX_BODIES * 4L)
    val ssboLocks    = ssbo(8, Cfg.MAX_BODIES * 4L)
    val ssboScanTmp  = ssbo(9,  (Cfg.MAX_BODIES / Cfg.LOCAL_SIZE + 2) * 4L)
    val ssboNewCount = ssbo(10, 4L)

    // init data
    initBodies(ssboPerm, ssboBodies, Cfg.START_BODIES)

    // установить начальный bodyCount в Globals (это не «физика», а размер массива)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboGlobals)
    val gbuf = glMapBufferRange(
        GL_SHADER_STORAGE_BUFFER, 0, 4L * 3 + 8L * 10,
        GL_MAP_WRITE_BIT or GL_MAP_INVALIDATE_RANGE_BIT
    )!!.order(java.nio.ByteOrder.nativeOrder())
    gbuf.asIntBuffer().put(1, Cfg.START_BODIES) // offset 1 = bodyCount
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    // clear victimFlags/locks перед первым кадром
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVictims)
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, null as java.nio.ByteBuffer?)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLocks)
    glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, null as java.nio.ByteBuffer?)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    // programs
    fun progCS(src: String) = linkProgram(intArrayOf(compile(GL_COMPUTE_SHADER, src)))
    fun progDraw(vs: String, fs: String) = linkProgram(intArrayOf(
        compile(GL_VERTEX_SHADER, vs), compile(GL_FRAGMENT_SHADER, fs)
    ))
    val pReset   = progCS(inject(csReset))
    val pBuild   = progCS(inject(csBuild))
    val pMass    = progCS(inject(csMass))
    val pAccel   = progCS(inject(csAccel))
    val pKick    = progCS(inject(csKick))
    val pDrift   = progCS(inject(csDrift))
    val pMerge   = progCS(inject(csMerge))
    val pCompact = progCS(inject(csCompact))
    val pDraw    = progDraw(inject(vsPoints), fsPoints)

    // geometry
    var drawCount = Cfg.START_BODIES
    val vao = vaoForDraw(Cfg.MAX_BODIES)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_PROGRAM_POINT_SIZE) // важный флаг для размера точек

    // uniforms
    val uWinReset   = glGetUniformLocation(pReset, "uWindow")
    val uLevelMass  = glGetUniformLocation(pMass, "uLevel")
    val uViewport   = glGetUniformLocation(pDraw, "uViewport")

    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents()

        // === STEP ===
        // 0) reset root + константы (bodyCount не трогаем)
        glUseProgram(pReset)
        glUniform2d(uWinReset, Cfg.WIDTH.toDouble(), Cfg.HEIGHT.toDouble())
        glDispatchCompute(1,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        var g = groups(drawCount)

        // 1) build
        glUseProgram(pBuild)
        glDispatchCompute(g,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // 2) mass bottom-up (фикс. максимум уровней)
        glUseProgram(pMass)
        for (lvl in 24 downTo 0) {
            glUniform1i(uLevelMass, lvl)
            glDispatchCompute(groups(Cfg.MAX_NODES),1,1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        }

        // 3) accel @ t
        glUseProgram(pAccel)
        glDispatchCompute(g,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // 4) kick half
        glUseProgram(pKick)
        glDispatchCompute(g,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // 5) drift
        glUseProgram(pDrift)
        glDispatchCompute(g,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // 6) rebuild for t+dt
        glUseProgram(pReset)
        glUniform2d(uWinReset, Cfg.WIDTH.toDouble(), Cfg.HEIGHT.toDouble())
        glDispatchCompute(1,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glUseProgram(pBuild)
        glDispatchCompute(g,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        glUseProgram(pMass)
        for (lvl in 24 downTo 0) {
            glUniform1i(uLevelMass, lvl)
            glDispatchCompute(groups(Cfg.MAX_NODES),1,1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        }

        // 7) accel @ t+dt
        glUseProgram(pAccel)
        glDispatchCompute(g,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // 8) kick half
        glUseProgram(pKick)
        glDispatchCompute(g,1,1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        // 9) merge + compaction
        if (Cfg.MERGE_MIN_DIST > 0.0) {
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboVictims)
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, null as java.nio.ByteBuffer?)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboLocks)
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32I, GL_RED_INTEGER, GL_INT, null as java.nio.ByteBuffer?)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            glUseProgram(pMerge)
            glDispatchCompute(g,1,1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            glUseProgram(pCompact)
            glDispatchCompute(g,1,1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

            // Обновим drawCount из Globals (только для количества точек)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssboGlobals)
            val map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 64, GL_MAP_READ_BIT)
            map?.order(java.nio.ByteOrder.nativeOrder())
            val bodyCount = map?.asIntBuffer()?.get(1) ?: drawCount
            glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)
            drawCount = bodyCount
            g = groups(drawCount)
        }

        // === RENDER ===
        glViewport(0,0, Cfg.WIDTH, Cfg.HEIGHT)
        glClearColor(0f,0f,0f,1f)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(pDraw)
        glUniform2f(uViewport, Cfg.WIDTH.toFloat(), Cfg.HEIGHT.toFloat())
        glBindVertexArray(vao)
        glDrawArrays(GL_POINTS, 0, drawCount)
        glBindVertexArray(0)

        glfwSwapBuffers(win)
    }

    glfwTerminate()
}
