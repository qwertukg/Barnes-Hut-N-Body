package kz.qwertukg.barneshut.shared

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

/**
 * Снимок тела, отправляемый клиенту.
 */
@Serializable
data class BodySnapshot(
    val x: Double,
    val y: Double,
    val mass: Double
)

/**
 * Служебная информация для HUD.
 */
@Serializable
data class HudInfo(
    val bodiesCount: Int,
    val paused: Boolean
)

/**
 * Пакет данных, приходящий на фронтенд.
 */
@Serializable
data class ServerFrame(
    val bodies: List<BodySnapshot>,
    val settings: SimulationSettings,
    val hud: HudInfo
)

/**
 * Сообщения, которые браузер отправляет на сервер.
 */
@Serializable
sealed class ClientCommand {
    @Serializable
    @SerialName("togglePause")
    data object TogglePause : ClientCommand()

    @Serializable
    @SerialName("adjustTheta")
    data class AdjustTheta(val delta: Double) : ClientCommand()

    @Serializable
    @SerialName("adjustBodies")
    data class AdjustBodies(val delta: Int) : ClientCommand()

    @Serializable
    @SerialName("adjustRadius")
    data class AdjustRadius(val delta: Double) : ClientCommand()

    @Serializable
    @SerialName("adjustDt")
    data class AdjustDt(val delta: Double) : ClientCommand()

    @Serializable
    @SerialName("adjustGravity")
    data class AdjustGravity(val delta: Double) : ClientCommand()

    @Serializable
    @SerialName("reset")
    data object Reset : ClientCommand()

    @Serializable
    @SerialName("clear")
    data object Clear : ClientCommand()

    @Serializable
    @SerialName("addDisk")
    data class AddDisk(
        val x: Double,
        val y: Double,
        val radius: Double,
        val count: Int,
        val vx: Double,
        val vy: Double,
        val clockwise: Boolean
    ) : ClientCommand()

    @Serializable
    @SerialName("updateViewport")
    data class UpdateViewport(val width: Int, val height: Int) : ClientCommand()
}
