import org.gradle.api.JavaVersion

plugins {
    kotlin("multiplatform")
    id("com.android.library")
}

kotlin {
    androidTarget()
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "17"
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")
            }
        }
        val androidMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
            }
        }
        val jvmMain by getting
    }
}

android {
    namespace = "com.example.barneshut.shared"
    compileSdk = 34

    defaultConfig {
        minSdk = 24
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
