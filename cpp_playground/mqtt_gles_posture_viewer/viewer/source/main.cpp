#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <thread>

void mqtt_main();

static const char *vertexShaderSource = R"(
#version 310 es
layout (location = 0) in vec3 aPos;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
}
)";

static const char *fragmentShaderSource = R"(
#version 310 es
out vec4 FragColor;
void main()
{
    FragColor = vec4(1.0f, 0.5f, 0.0f, 1.0f);
}
)";

static float vertices[] = {
    -0.5f, -0.5f, -0.5f,  0.5f, -0.5f, -0.5f,  // 边1
     0.5f, -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  // 边2
     0.5f,  0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  // 边3
    -0.5f,  0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  // 边4
    -0.5f, -0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  // 边5
     0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  0.5f,  // 边6
     0.5f,  0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  // 边7
    -0.5f,  0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  // 边8
    -0.5f, -0.5f, -0.5f, -0.5f, -0.5f,  0.5f,  // 边9
     0.5f, -0.5f, -0.5f,  0.5f, -0.5f,  0.5f,  // 边10
     0.5f,  0.5f, -0.5f,  0.5f,  0.5f,  0.5f,  // 边11
    -0.5f,  0.5f, -0.5f, -0.5f,  0.5f,  0.5f   // 边12
};

static void processInput(GLFWwindow *window)
{
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

static unsigned int compileShader(unsigned int type, const std::string& source)
{
    unsigned int id = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(id, 1, &src, nullptr);
    glCompileShader(id);

    int result;
    glGetShaderiv(id, GL_COMPILE_STATUS, &result);
    if (result == GL_FALSE)
    {
        int length;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
        char* message = (char*)alloca(length * sizeof(char));
        glGetShaderInfoLog(id, length, &length, message);
        std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
        std::cout << message << std::endl;
        glDeleteShader(id);
        return 0;
    }

    return id;
}

static unsigned int createShader(const std::string& vertexShader, const std::string& fragmentShader)
{
    unsigned int program = glCreateProgram();
    unsigned int vs = compileShader(GL_VERTEX_SHADER, vertexShader);
    unsigned int fs = compileShader(GL_FRAGMENT_SHADER, fragmentShader);

    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);
    glValidateProgram(program);

    glDeleteShader(vs);
    glDeleteShader(fs);

    return program;
}

float euler[3];

int main()
{
    std::thread(mqtt_main).detach();

    glfwInit();
    GLFWwindow* window = glfwCreateWindow(800, 600, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (!gladLoadGLES2Loader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    unsigned int shader = createShader(vertexShaderSource, fragmentShaderSource);
    glUseProgram(shader);

    glViewport(0, 0, 800, 600);
    //glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // 创建顶点缓冲对象和顶点数组对象
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    // 绑定和设置顶点缓冲和数组
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shader);
        glBindVertexArray(VAO);

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, glm::radians(euler[0]), glm::vec3(1.0f, 0.0f,0.0f));
        model = glm::rotate(model, glm::radians(euler[1]), glm::vec3(0.0f, 1.0f,0.0f));
        model = glm::rotate(model, glm::radians(euler[2]), glm::vec3(0.0f, 0.0f,1.0f));
        unsigned int modelLoc = glGetUniformLocation(shader, "model");
        glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

        glm::mat4 view = glm::mat4(1.0f);
        view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
        unsigned int viewLoc = glGetUniformLocation(shader, "view");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));

        glm::mat4 projection = glm::mat4(1.0f);
        projection = glm::perspective(glm::radians(45.0f), 800.0f / 600.0f, 0.1f, 100.0f);
        unsigned int projLoc = glGetUniformLocation(shader, "projection");
        glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(projection));

        glDrawArrays(GL_LINES, 0, 48);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}