plugins {
    id("java")
}

group = "me.iaksh"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.test {
    useJUnitPlatform()
}

tasks.jar {
    // enabled = true
    manifest {
        attributes(mapOf("Main-Class" to "me.iaksh.hexagram.Main"))
    }
}

tasks.withType<JavaExec> {
    systemProperty("file.encoding", "utf-8")
}

tasks.withType<JavaCompile> {
    options.encoding = "utf-8"
}
