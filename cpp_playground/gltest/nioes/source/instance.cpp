#include "nioes/instance.hpp"

static size_t instance_count{ 0 };
static bool glfw_loaded{ false };

nioes::Instance::Instance(std::string_view title,int w,int h) noexcept(false) {
    instance_count++;
    create_glfw_context(title,w,h);
}

nioes::Instance::~Instance() noexcept {
    glfwDestroyWindow(window);
    if(!(--instance_count)) {
        glfwTerminate();
    }
}

void nioes::Instance::create_glfw_context(std::string_view title,int w,int h) noexcept(false) {
    // 如果没有任何一个instance存在，则GLFW以及OpenGL环境不存在
    // 尝试创建环境并增加instance计数
    bool need_create_context{ false };
    if(!(instance_count++)) {
        need_create_context = true;

        glfwInit();
        glfwWindowHint(GLFW_CLIENT_API,GLFW_OPENGL_ES_API);
        glfwWindowHint(GLFW_CONTEXT_CREATION_API,GLFW_NATIVE_CONTEXT_API);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,2);
        glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);

        glfwSetErrorCallback([](int error,const char* description) {
            spdlog::critical("GLFW error: {}",description);
        });
    }

    // 创建glfw窗口
    window = glfwCreateWindow(w,h,title.data(),nullptr,nullptr);
    glfwMakeContextCurrent(window);

    // 获取OpenGL ES API
    if(need_create_context) {
        int version = gladLoadGLES2(glfwGetProcAddress);
        if(!version) {
            throw std::runtime_error("can't find GLES2 API address");
        }

        spdlog::info("Loaded OpenGL ES {}.{}\nGL_VENDOR\t{}\nGL_RENDERER\t{}\nGL_VERSION\t{}",
            GLAD_VERSION_MAJOR(version),
            GLAD_VERSION_MINOR(version),
            reinterpret_cast<const char*>(glGetString(GL_VENDOR)),
            reinterpret_cast<const char*>(glGetString(GL_RENDERER)),
            reinterpret_cast<const char*>(glGetString(GL_VERSION))
        );
    }

    // GLFW有一个远古BUG，在创建好窗口后会立即生成一个OpenGL Error
    // 在此处直接捕获以防止其在其他地方突然发作
    glGetError();
}

void nioes::Instance::flush() noexcept {
    glfwSwapBuffers(window);
}

void nioes::Instance::update_all() noexcept(false) {
    if(!instance_count) {
        throw std::runtime_error("no gl instance ot update");
    }
    glfwPollEvents();
}