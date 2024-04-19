#pragma once

#include <jumping_ball/physics.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace jumping_ball::graphics {
	inline static GLenum glCheckError_(const char* file, int line)
	{
		GLenum error_code;
		while ((error_code = glGetError()) != GL_NO_ERROR)
		{
			std::string error;
			switch (error_code)
			{
			case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
			case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
			case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
			case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
			case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
			case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
			case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
			}
			spdlog::error("{} | {} ({})", error, file, line);
		}
		return error_code;
	}
#define glCheckError() glCheckError_(__FILE__, __LINE__)

	extern GLFWwindow* window;

	void initGraphics() noexcept;
	void closeGraphics() noexcept;

	class RenPipe {
	public:
		GLuint vao_id, vbo_id, ebo_id;
		GLuint shader_id;

		RenPipe(const std::string_view& vshader_source, const std::string_view& fshader_source,
			const std::vector<float>& vertices, const std::vector<unsigned int>& indices) noexcept;
		~RenPipe() noexcept;

		void updateVertices(const std::vector<float>& vertices, const std::vector<unsigned int>& indices) noexcept;
		// TODO: 这个函数可能需要很多参数
		// TODO: 临时保留为绘制圆的函数
		virtual void draw(const glm::vec3& position,float r) noexcept;

	private:
		void initialize(const std::string_view& vshader_source, const std::string_view& fshader_source) noexcept;
		void uninitiaze() noexcept;
	};
}