plugins {
    kotlin("jvm") version "2.2.20"
    application
}

group = "kz.qwertukg"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.9.0")
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(18)
}

application {
    mainClass.set("MainKt")
}

tasks.test {
    useJUnitPlatform()
}