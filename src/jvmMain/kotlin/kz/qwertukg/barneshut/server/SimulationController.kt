package kz.qwertukg.barneshut.server

import io.ktor.util.logging.KtorSimpleLogger
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeoutOrNull
import kz.qwertukg.barneshut.shared.ClientCommand
import kz.qwertukg.barneshut.shared.HudInfo
import kz.qwertukg.barneshut.shared.ServerFrame
import kz.qwertukg.barneshut.shared.SimulationConstants
import kz.qwertukg.barneshut.shared.SimulationSettings
import kz.qwertukg.barneshut.simulation.BodyFactory
import kz.qwertukg.barneshut.simulation.PhysicsEngine
import kz.qwertukg.barneshut.simulation.toSnapshot

/** Управляет состоянием симуляции и общением с клиентами. */
class SimulationController {
    private val logger = KtorSimpleLogger("Simulation")
    private val scope = CoroutineScope(Dispatchers.Default)
    private val broadcaster = MutableSharedFlow<ServerFrame>(replay = 1, extraBufferCapacity = 1)
    private val commands = Channel<ClientCommand>(Channel.UNLIMITED)

    private var settings: SimulationSettings = SimulationSettings()
    private var paused: Boolean = false
    private val engine = PhysicsEngine(BodyFactory.defaultBodies(settings), settings)

    init {
        scope.launch {
            broadcastSnapshot()
            simulationLoop()
        }
    }

    /** Поток кадров для клиентов. */
    val frames: SharedFlow<ServerFrame> = broadcaster.asSharedFlow()

    /** Передать команду от клиента. */
    suspend fun submit(command: ClientCommand) {
        commands.send(command)
    }

    private suspend fun simulationLoop() {
        while (scope.isActive) {
            handlePendingCommands()
            if (!paused) {
                try {
                    engine.step()
                } catch (t: Throwable) {
                    logger.error("Ошибка шага симуляции", t)
                    paused = true
                }
            }
            broadcastSnapshot()
            delay(16L)
        }
    }

    private suspend fun handlePendingCommands() {
        while (true) {
            val command = withTimeoutOrNull(0L) { commands.receiveCatching().getOrNull() } ?: break
            when (command) {
                ClientCommand.TogglePause -> paused = !paused
                is ClientCommand.AdjustTheta -> updateSettings {
                    it.copy(theta = (it.theta + command.delta).coerceIn(SimulationConstants.THETA_MIN, SimulationConstants.THETA_MAX))
                }
                is ClientCommand.AdjustBodies -> updateSettings {
                    it.copy(bodiesPerDisk = (it.bodiesPerDisk + command.delta).coerceIn(
                        SimulationConstants.DISK_BODIES_MIN,
                        SimulationConstants.DISK_BODIES_MAX
                    ))
                }
                is ClientCommand.AdjustRadius -> updateSettings {
                    it.copy(diskRadius = (it.diskRadius + command.delta).coerceIn(
                        SimulationConstants.DISK_RADIUS_MIN,
                        SimulationConstants.DISK_RADIUS_MAX
                    ))
                }
                is ClientCommand.AdjustDt -> updateSettings {
                    it.copy(timeStep = (it.timeStep + command.delta).coerceIn(
                        SimulationConstants.DT_MIN,
                        SimulationConstants.DT_MAX
                    ))
                }
                is ClientCommand.AdjustGravity -> updateSettings {
                    it.copy(gravity = (it.gravity + command.delta).coerceIn(
                        SimulationConstants.GRAVITY_MIN,
                        SimulationConstants.GRAVITY_MAX
                    ))
                }
                ClientCommand.Reset -> engine.resetBodies(BodyFactory.defaultBodies(settings))
                ClientCommand.Clear -> engine.resetBodies(mutableListOf())
                is ClientCommand.AddDisk -> {
                    val radius = command.radius.coerceIn(
                        SimulationConstants.DISK_RADIUS_MIN,
                        SimulationConstants.DISK_RADIUS_MAX
                    )
                    val count = command.count.coerceIn(
                        SimulationConstants.DISK_BODIES_MIN,
                        SimulationConstants.DISK_BODIES_MAX
                    )
                    val disk = BodyFactory.makeKeplerDisk(
                        settings = settings,
                        nTotal = count,
                        vx = command.vx,
                        vy = command.vy,
                        x = command.x,
                        y = command.y,
                        r = radius,
                        clockwise = command.clockwise
                    )
                    engine.resetBodies((engine.getBodies() + disk).toMutableList())
                }
                is ClientCommand.UpdateViewport -> updateSettings {
                    it.copy(
                        widthPx = command.width.coerceAtLeast(100),
                        heightPx = command.height.coerceAtLeast(100)
                    )
                }
            }
        }
    }

    private fun updateSettings(transform: (SimulationSettings) -> SimulationSettings) {
        val updated = transform(settings)
        if (updated != settings) {
            settings = updated
            engine.updateSettings(settings)
        }
    }

    private suspend fun broadcastSnapshot() {
        val bodies = engine.getBodies()
        val frame = ServerFrame(
            bodies = bodies.map { it.toSnapshot() },
            settings = settings,
            hud = HudInfo(bodies.size, paused)
        )
        broadcaster.emit(frame)
    }
}
