import org.gradle.api.tasks.Copy
import org.gradle.api.tasks.JavaExec
import org.jetbrains.kotlin.gradle.targets.js.webpack.KotlinWebpack

plugins {
    kotlin("multiplatform") version "2.2.20"
    id("org.jetbrains.kotlin.plugin.serialization") version "2.2.20"
}

group = "kz.qwertukg"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

kotlin {
    jvm()
    js(IR) {
        browser {
            binaries.executable()
            commonWebpackConfig {
                outputFileName = "app.js"   // ← фиксируем имя
            }
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
            }
        }
        val jvmMain by getting {
            dependencies {
                implementation("io.ktor:ktor-server-core:3.0.0")
                implementation("io.ktor:ktor-server-netty:3.0.0")
                implementation("io.ktor:ktor-server-content-negotiation:3.0.0")
                implementation("io.ktor:ktor-server-websockets:3.0.0")
                implementation("io.ktor:ktor-serialization-kotlinx-json:3.0.0")
                implementation("ch.qos.logback:logback-classic:1.5.12")
            }
        }
        val jvmTest by getting
        val jsMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
            }
        }
        val jsTest by getting
    }

    jvmToolchain(18)
}

val jsBrowserProductionWebpack by tasks.existing(KotlinWebpack::class)

tasks.named<Copy>("jvmProcessResources") {
    dependsOn(jsBrowserProductionWebpack)

    // копируем бандлы JS
    from(jsBrowserProductionWebpack.flatMap { it.outputDirectory }) {
        into("static")
    }

    // копируем index.html из jsMain в jvm-ресурсы
    from("src/jsMain/resources/index.html") {
        into("static")
    }
}

tasks.register<JavaExec>("runServer") {
    group = "application"
    description = "Запустить Ktor-сервер Barnes–Hut"
    dependsOn("jvmJar")
    mainClass.set("kz.qwertukg.barneshut.server.ServerKt")
    val runtimeClasspath = configurations.named("jvmRuntimeClasspath")
    classpath = files(
        kotlin.targets["jvm"].compilations["main"].output.allOutputs,
        runtimeClasspath.get()
    )
}
