// GLFW和OpenGL状态机
// Key/Mouse Input以及OpenGL状态维护
#include <glad/gles2.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace nioes {
	// Instance实际上就是窗口，不过共用同一个OpenGL状态机
	class Instance {
	private:
		GLFWwindow* window;
		void create_glfw_context(std::string_view title,int w,int h) noexcept;

	public:
		Instance(std::string_view title,int w,int h) noexcept;
		Instance(Instance&) = delete;
		~Instance() noexcept;
		void flush() noexcept;
		static void update_all() noexcept;
	};
}