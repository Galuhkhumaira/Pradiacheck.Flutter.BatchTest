allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

// Memaksa SEMUA subproject (termasuk plugin pihak ketiga seperti
// tflite_flutter dan onnxruntime) memakai pengaturan yang sama:
// - Java & Kotlin target 17 (menghindari error "Inconsistent JVM Target
//   Compatibility Between Java and Kotlin Tasks").
// - compileSdk 36 (onnxruntime di-compile dengan android-33, sedangkan
//   library AndroidX yang dipakainya butuh minimal 34).
// Blok ini harus berada SEBELUM evaluationDependsOn(":app").
subprojects {
    afterEvaluate {
        extensions.findByType<com.android.build.gradle.BaseExtension>()?.apply {
            compileSdkVersion(36)
            compileOptions {
                sourceCompatibility = JavaVersion.VERSION_17
                targetCompatibility = JavaVersion.VERSION_17
            }
        }
        tasks.withType<org.jetbrains.kotlin.gradle.tasks.KotlinCompile>().configureEach {
            compilerOptions {
                jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
            }
        }
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}