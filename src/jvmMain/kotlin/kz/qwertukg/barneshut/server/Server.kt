package kz.qwertukg.barneshut.server

import io.ktor.http.ContentType
import io.ktor.serialization.kotlinx.KotlinxWebsocketSerializationConverter
import io.ktor.server.application.call
import io.ktor.server.application.install
import io.ktor.server.engine.embeddedServer
import io.ktor.server.http.content.staticResources
import io.ktor.server.netty.Netty
import io.ktor.server.response.respondBytes
import io.ktor.server.response.respondRedirect
import io.ktor.server.routing.get
import io.ktor.server.routing.routing
import io.ktor.server.websocket.WebSockets
import io.ktor.server.websocket.sendSerialized
import io.ktor.server.websocket.webSocket
import io.ktor.websocket.Frame
import io.ktor.websocket.readText
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect
import kotlinx.serialization.json.Json
import kz.qwertukg.barneshut.shared.ClientCommand
import kz.qwertukg.barneshut.shared.ServerFrame
import kotlin.io.use

/** Точка входа серверного приложения. */
fun main() {
    val controller = SimulationController()
    val port = System.getenv("PORT")?.toIntOrNull() ?: 8080
    val json = Json {
        ignoreUnknownKeys = true
        encodeDefaults = true
        classDiscriminator = "type"
    }

    embeddedServer(Netty, port = port) {
        install(WebSockets) {
            contentConverter = KotlinxWebsocketSerializationConverter(json)
        }

        routing {
            get("/") {
                val resource = call.application.environment.classLoader.getResource("static/index.html")
                if (resource != null) {
                    val bytes = resource.openStream().use { it.readBytes() }
                    call.respondBytes(bytes, contentType = ContentType.Text.Html)
                } else {
                    call.respondRedirect("/static/index.html")
                }
            }

            staticResources("/static", "static")

            webSocket("/ws") {
                val sender = launch {
                    controller.frames.collect { frame: ServerFrame ->
                        sendSerialized(frame)
                    }
                }
                try {
                    for (frame in incoming) {
                        when (frame) {
                            is Frame.Text -> {
                                val command = json.decodeFromString(ClientCommand.serializer(), frame.readText())
                                controller.submit(command)
                            }
                            else -> {}
                        }
                    }
                } finally {
                    sender.cancel()
                }
            }
        }
    }.start(wait = true)
}
