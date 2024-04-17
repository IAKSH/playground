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

	class BallRenObject {
	private:
		void initShader(const std::string& vshader_source, const std::string& fshader_source);
		void loadCircleVertices();

	public:
		GLuint vao_id, vbo_id, ebo_id;
		GLuint shader_id;

		BallRenObject(const std::string& vertex_shader, const std::string& fragment_shader);
		~BallRenObject() noexcept;

		void draw(const physics::Ball& ball);
	};
}