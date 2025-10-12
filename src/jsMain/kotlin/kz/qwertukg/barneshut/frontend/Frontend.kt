package kz.qwertukg.barneshut.frontend

import kotlinx.browser.document
import kotlinx.browser.window
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import kz.qwertukg.barneshut.shared.ClientCommand
import kz.qwertukg.barneshut.shared.ServerFrame
import org.w3c.dom.HTMLCanvasElement
import org.w3c.dom.HTMLDivElement
import org.w3c.dom.MessageEvent
import org.w3c.dom.WebSocket
import org.w3c.dom.events.Event
import org.w3c.dom.events.KeyboardEvent
import org.w3c.dom.events.MouseEvent
import org.w3c.dom.CanvasRenderingContext2D
import kotlin.math.PI

/** Коэффициент перевода пикселей перетаскивания в скорость. */
private const val VEL_PER_PIXEL = 1.0

private val json = Json {
    ignoreUnknownKeys = true
    encodeDefaults = true
    classDiscriminator = "type"
}

private var currentFrame: ServerFrame? = null
private var socket: WebSocket? = null
private lateinit var canvas: HTMLCanvasElement
private lateinit var ctx: CanvasRenderingContext2D
private lateinit var hud: HTMLDivElement

private var dragStartX: Double? = null
private var dragStartY: Double? = null
private var dragCurrentX: Double? = null
private var dragCurrentY: Double? = null

fun main() {
    setupDom()
    connectWebSocket()
    window.addEventListener("resize", { resizeCanvas() })
}

private fun setupDom() {
    document.body?.style?.backgroundColor = "black"
    canvas = (document.createElement("canvas") as HTMLCanvasElement).apply {
        style.width = "100vw"
        style.height = "100vh"
        style.display = "block"
    }
    document.body?.appendChild(canvas)
    ctx = canvas.getContext("2d") as CanvasRenderingContext2D

    hud = (document.createElement("div") as HTMLDivElement).apply {
        style.position = "fixed"
        style.left = "16px"
        style.top = "16px"
        style.color = "#00ff00"
        style.fontFamily = "monospace"
        style.fontSize = "14px"
        style.whiteSpace = "pre"
        style.asDynamic().pointerEvents = "none"
        textContent = "Подключение к симуляции..."
    }
    document.body?.appendChild(hud)

    document.addEventListener("keydown", ::handleKey)
    canvas.addEventListener("mousedown", ::handleMouseDown)
    canvas.addEventListener("mousemove", ::handleMouseMove)
    canvas.addEventListener("mouseup", ::handleMouseUp)
    canvas.addEventListener("mouseleave", ::handleMouseLeave)
    canvas.addEventListener("contextmenu", { event: Event -> event.preventDefault() })
    canvas.addEventListener("wheel", { event: Event -> event.preventDefault() })

    resizeCanvas()
}

private fun resizeCanvas() {
    val w = window.innerWidth
    val h = window.innerHeight
    canvas.width = w
    canvas.height = h
    sendViewport()
}

private fun connectWebSocket() {
    val origin = window.location.origin.replace("http", "ws")
    val ws = WebSocket("$origin/ws")
    socket = ws

    ws.onopen = {
        sendViewport()
        hud.textContent = "Соединение установлено, ожидание данных..."
    }

    ws.onmessage = fun(event: MessageEvent) {
        val data = event.data as? String ?: return
        val frame = json.decodeFromString(ServerFrame.serializer(), data)
        currentFrame = frame
        hud.textContent = buildHud(frame)
        requestRedraw()
    }

    ws.onclose = {
        hud.textContent = "Связь потеряна. Переподключение..."
        window.setTimeout({ connectWebSocket() }, 1_000)
    }

    ws.onerror = {
        hud.textContent = "Ошибка WebSocket"
    }
}

private fun buildHud(frame: ServerFrame): String {
    val settings = frame.settings
    return buildString {
        appendLine("SPACE — пауза | R — сброс | MOUSE1 — диск | MOUSE2 — очистка")
        appendLine("Диск: радиус ${settings.diskRadius} | тела ${settings.bodiesPerDisk}")
        appendLine("Theta ${settings.theta} | dt ${settings.timeStep} | G ${settings.gravity}")
        append("Всего тел: ${frame.hud.bodiesCount} | Пауза: ${frame.hud.paused}")
    }
}

private var redrawScheduled = false

private fun requestRedraw() {
    if (redrawScheduled) return
    redrawScheduled = true
    window.requestAnimationFrame {
        redrawScheduled = false
        render()
    }
}

private fun render() {
    val frame = currentFrame ?: return
    ctx.fillStyle = "black"
    ctx.fillRect(0.0, 0.0, canvas.width.toDouble(), canvas.height.toDouble())

    ctx.fillStyle = "white"
    val bodies = frame.bodies
    for (body in bodies) {
        val x = body.x
        val y = body.y
        if (x >= 0 && x < canvas.width && y >= 0 && y < canvas.height) {
            ctx.fillRect(x, y, 1.0, 1.0)
        }
    }

    drawDragPreview()
}

private fun drawDragPreview() {
    val sx = dragStartX ?: return
    val sy = dragStartY ?: return
    val cx = dragCurrentX ?: sx
    val cy = dragCurrentY ?: sy

    ctx.strokeStyle = "rgba(0,255,0,0.8)"
    ctx.setLineDash(arrayOf(6.0, 6.0))
    ctx.beginPath()
    ctx.moveTo(sx, sy)
    ctx.lineTo(cx, cy)
    ctx.stroke()

    val radius = currentFrame?.settings?.diskRadius ?: 0.0
    ctx.beginPath()
    ctx.setLineDash(arrayOf(4.0, 4.0))
    ctx.arc(sx, sy, radius, 0.0, 2 * PI)
    ctx.stroke()
    ctx.setLineDash(emptyArray<Double>())
}

private fun handleKey(event: dynamic) {
    val e = event as KeyboardEvent
    val command = when (e.code) {
        "Space" -> ClientCommand.TogglePause
        "KeyZ" -> ClientCommand.AdjustTheta(-0.05)
        "KeyX" -> ClientCommand.AdjustTheta(0.05)
        "KeyA" -> ClientCommand.AdjustBodies(-100)
        "KeyS" -> ClientCommand.AdjustBodies(100)
        "KeyQ" -> ClientCommand.AdjustRadius(-10.0)
        "KeyW" -> ClientCommand.AdjustRadius(10.0)
        "KeyO" -> ClientCommand.AdjustDt(-0.001)
        "KeyP" -> ClientCommand.AdjustDt(0.001)
        "KeyK" -> ClientCommand.AdjustGravity(-1.0)
        "KeyL" -> ClientCommand.AdjustGravity(1.0)
        "KeyR" -> ClientCommand.Reset
        else -> null
    }
    if (command != null) {
        e.preventDefault()
        sendCommand(command)
    }
}

private fun handleMouseDown(event: dynamic) {
    val e = event as MouseEvent
    when (e.button.toInt()) {
        0 -> {
            dragStartX = e.offsetX
            dragStartY = e.offsetY
            dragCurrentX = e.offsetX
            dragCurrentY = e.offsetY
            requestRedraw()
        }
        1 -> {
            e.preventDefault()
            sendCommand(ClientCommand.Clear)
        }
    }
}

private fun handleMouseMove(event: dynamic) {
    val e = event as MouseEvent
    if (dragStartX != null) {
        dragCurrentX = e.offsetX
        dragCurrentY = e.offsetY
        requestRedraw()
    }
}

private fun handleMouseUp(event: dynamic) {
    val e = event as MouseEvent
    if (e.button.toInt() == 0 && dragStartX != null) {
        val sx = dragStartX!!
        val sy = dragStartY!!
        val ex = e.offsetX
        val ey = e.offsetY
        val vx = (ex - sx) * VEL_PER_PIXEL
        val vy = (ey - sy) * VEL_PER_PIXEL
        val settings = currentFrame?.settings
        if (settings != null) {
            sendCommand(
                ClientCommand.AddDisk(
                    x = sx,
                    y = sy,
                    radius = settings.diskRadius,
                    count = settings.bodiesPerDisk,
                    vx = vx,
                    vy = vy,
                    clockwise = true
                )
            )
        }
        dragStartX = null
        dragStartY = null
        dragCurrentX = null
        dragCurrentY = null
        requestRedraw()
    }
}

private fun handleMouseLeave(@Suppress("UNUSED_PARAMETER") event: dynamic) {
    dragStartX = null
    dragStartY = null
    dragCurrentX = null
    dragCurrentY = null
    requestRedraw()
}

private fun sendViewport() {
    val width = canvas.width
    val height = canvas.height
    sendCommand(ClientCommand.UpdateViewport(width, height))
}

private fun sendCommand(command: ClientCommand) {
    val ws = socket ?: return
    if (ws.readyState != WebSocket.OPEN) return
    ws.send(json.encodeToString(ClientCommand.serializer(), command))
}
