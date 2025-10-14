# Barnes–Hut N-Body

![collide](collide.gif)

Visualization of gravitational dynamics using the Barnes–Hut algorithm implemented in Kotlin/Swing. The application launches in a fullscreen borderless mode, automatically adapts to the current display resolution, and demonstrates the collision of two Keplerian disks composed of a central massive star and a ring of satellites with correctly computed enclosed mass.

## Quick Start

1. Install JDK 18 (Temurin or any compatible distribution will do).
2. Build the project via Gradle Wrapper:
   ```bash
   ./gradlew run
   ```
   You can also import the project into IntelliJ IDEA and run `main()` from `Main.kt`.

## Algorithm and Implementation Details

* Space is partitioned by the `BHTree` quadtree; the approximation criterion is `s² < θ² · dist²`, where θ adapts on the fly (`0.2`–`1.6`).
* Force calculations run in parallel via `kotlinx.coroutines` on as many threads as there are available CPU cores.
* Integration uses symmetric Leapfrog (kick–drift–kick) with an adjustable step `Δt`.
* Force softening (`ε = 1.0`) is applied to stabilize close encounters.

## Controls

### Mouse
* **LMB + drag** — create a new Keplerian disk at the click position; the drag vector defines the additional velocity (`1 px = 1 unit/s`).
* **Middle button** — clear the scene.
* **ARROWS** — camera movement.

### Keyboard
* `SPACE` — pause/resume.
* `R` — reset to the initial two-disk configuration.
* `ESC` — exit.
* `Z / X` — decrease/increase θ (Barnes–Hut).
* `A / S` — change the number of bodies in the added disk (1,000–10,000, step 100).
* `Q / W` — change the radius of the created disk (100–500 px, step 10).
* `O / P` — decrease/increase the integration step `Δt` (from −0.015 to 0.015).
* `K / L` — decrease/increase the gravitational constant `G` (0–100).
* `WHEEL DOWN/UP` — decrease/increase zoom (x1–x10).
* `D` — toggle debug mode.

## HUD and Diagnostics

The upper-left corner displays a hint line for the main shortcuts, the current values of `R`, `N`, `θ`, `Δt`, `G`, and the system size (number of particles). For debugging, a dashed line is drawn from the mouse click and a circle shows the radius of the future disk.

## Configuration

Additional parameters (window size, softening magnitude, masses, etc.) are centralized in `Config.kt` and can be adjusted before building.
