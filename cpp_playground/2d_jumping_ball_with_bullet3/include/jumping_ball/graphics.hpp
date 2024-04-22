#pragma once

#include <jumping_ball/physics.hpp>
#include <jumping_ball/misc.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <spdlog/spdlog.h>

namespace jumping_ball::graphics {
	using namespace misc;

	extern GLFWwindow* window;

	void initialize() noexcept;
	void uninitialize() noexcept;

	/*
	+ Camera (O)

	+ RenObject (√)
		+ vao (√)
		+ vbo (√)
		+ ebo (√)
		+ indices_cnt (√)

	+ GLSLPreprocessor (O)

	+ Renderer (O)
		+ RenPipe -> Frame (or RenderPass) (√)
			+ shader (√)
	*/

	class Camera {
    public:
		RotatablePoint rotatable_point;
        int screen_width;
        int screen_height;
		bool enable_ortho;

        Camera(const float& w,const float& h,const float& fov = 45.0f,const float& zoom = 1.0f) noexcept;
        ~Camera() = default;

        float getZoom() const noexcept;
        float getFOV() const noexcept;
		void setZoom(const float& val) noexcept;
        void setFov(const float& val) noexcept;
        glm::mat4 getMatrix() const noexcept;

	private:
		float fov;
        float zoom;
    };

	class RenObject {
	public:
		GLuint vao_id, vbo_id, ebo_id;
		unsigned int indices_cnt;
		const std::vector<float>& vertices;
		const std::vector<unsigned int>& indices;
		glm::vec3 position;
		glm::quat orientation;
		
		RenObject(const std::vector<float>& vertices, const std::vector<unsigned int>& indices) noexcept;
		RenObject() noexcept;
		~RenObject() noexcept;
		void updateVertices() noexcept;
		void updateVertices(const std::vector<float>& vertices, const std::vector<unsigned int>& indices) noexcept;

	private:
		void initialize() noexcept;
		void uninitialize() noexcept;
	};

	class RenPipe {
	public:
		GLuint shader_id;

		RenPipe(const std::string_view& vshader_source, const std::string_view& fshader_source) noexcept;
		~RenPipe() noexcept;

		void draw(RenObject& obj) noexcept;
		void setCamera(std::shared_ptr<Camera> camera) noexcept;

	private:
		void initialize(const std::string_view& vshader_source, const std::string_view& fshader_source) noexcept;
		void uninitiaze() noexcept;
		std::shared_ptr<Camera> camera;
	};

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
}