#include <jumping_ball/graphics.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/gtx/quaternion.hpp>

GLFWwindow* jumping_ball::graphics::window;

namespace jumping_ball::graphics {
	void initialize() noexcept {
		glfwInit();
		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

		window = glfwCreateWindow(800, 800, "2D Jumping Ball", nullptr, nullptr);
		if (!window) {
			spdlog::critical("failed to create GLFW window");
			//glfwTerminate();
			std::terminate();
		}

		glfwMakeContextCurrent(window);

		glfwSwapInterval(1);

		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
			spdlog::critical("failed to initialize GLAD");
			std::terminate();
		}

		spdlog::info("OpenGL, Launch!");
		spdlog::info("OpenGL vendor:  \t{}", reinterpret_cast<const char*>(glGetString(GL_VENDOR)));
		spdlog::info("OpenGL renderer:\t{}", reinterpret_cast<const char*>(glGetString(GL_RENDERER)));
		spdlog::info("OpenGL version: \t{}", reinterpret_cast<const char*>(glGetString(GL_VERSION)));

		glViewport(0, 0, 800, 800);
		glfwSetFramebufferSizeCallback(window, [](GLFWwindow* window, int width, int height) noexcept
			{
				glViewport(0, 0, width, height);
			});
	}

	void uninitialize() noexcept {
		glfwTerminate();
	}

	RenObject::RenObject(const std::vector<float>& vertices, const std::vector<unsigned int>& indices) noexcept
		: vertices(vertices),indices(indices)
	{
		initialize();
	}

	RenObject::~RenObject() noexcept {
		uninitialize();
	}

	void RenObject::updateVertices(const std::vector<float>& vertices,const std::vector<unsigned int>& indices) noexcept {
		indices_cnt = indices.size();

		glBindVertexArray(vao_id);

		glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_id);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}

	void RenObject::updateVertices() noexcept {
		updateVertices(vertices,indices);
	}

	void RenObject::initialize() noexcept {
		glGenVertexArrays(1, &vao_id);
		glGenBuffers(1, &vbo_id);
		glGenBuffers(1, &ebo_id);
		updateVertices(vertices,indices);
	}

	void RenObject::uninitialize() noexcept {
		glDeleteVertexArrays(1, &vao_id);
		glDeleteBuffers(1, &vbo_id);
		glDeleteBuffers(1, &ebo_id);
	}

	RenPipe::RenPipe(const std::string_view& vshader_source, const std::string_view& fshader_source) noexcept {
		initialize(vshader_source, fshader_source);
	}

	RenPipe::~RenPipe() noexcept {
		uninitiaze();
	}

	void jumping_ball::graphics::RenPipe::draw(RenObject& obj) noexcept {
		// TODO: temp code

		// 计算变换矩阵
		// 暂时直接将窗口大小视为固定的800x800
		float window_scaled_r = 1.0f / 400;
		float window_scaled_x = obj.position.x / 400;
		float window_scaled_y = obj.position.y / 400;

		glm::mat4 mvp_matrix = 
			glm::translate(glm::mat4(1.0f), glm::vec3(window_scaled_x, window_scaled_y, obj.position.z))
			* glm::mat4_cast(obj.orientation)
			* glm::scale(glm::mat4(1.0f), glm::vec3(window_scaled_r, window_scaled_r, window_scaled_r))
			* glm::mat4(1.0f);
		if(camera) {
			mvp_matrix = camera->getMatrix() * mvp_matrix;
		}

		// 将变换矩阵传递给着色器
		glUseProgram(shader_id);
		glUniformMatrix4fv(glGetUniformLocation(shader_id, "mvp_matrix"), 1, GL_FALSE, glm::value_ptr(mvp_matrix));
		
		// 临时
		glEnable(GL_CULL_FACE);
		glFrontFace(GL_CW);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		//glLineWidth(5.0f);

		// 绘制圆形
		glBindVertexArray(obj.vao_id);
		glDrawElements(GL_TRIANGLES, obj.indices_cnt, GL_UNSIGNED_INT, 0);
		//glDrawElements(GL_TRIANGLE_FAN, indices_cnt, GL_UNSIGNED_INT, 0);
	}

	void RenPipe::setCamera(std::shared_ptr<Camera> camera) noexcept {
		this->camera = camera;
	}

	void RenPipe::initialize(const std::string_view& vshader_source,const std::string_view& fshader_source) noexcept {
		// 创建顶点着色器
		GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
		const char* vshader_source_cstr = vshader_source.data();
		glShaderSource(vertex_shader, 1, &vshader_source_cstr, NULL);
		glCompileShader(vertex_shader);

		// 创建片段着色器
		GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
		const char* fshader_source_cstr = fshader_source.data();
		glShaderSource(fragment_shader, 1, &fshader_source_cstr, NULL);
		glCompileShader(fragment_shader);

		// 创建着色器程序
		shader_id = glCreateProgram();
		glAttachShader(shader_id, vertex_shader);
		glAttachShader(shader_id, fragment_shader);
		glLinkProgram(shader_id);

		// 删除着色器
		glDeleteShader(vertex_shader);
		glDeleteShader(fragment_shader);
	}

	void jumping_ball::graphics::RenPipe::uninitiaze() noexcept {
		glDeleteProgram(shader_id);
	}

	Camera::Camera(const float& w,const float& h,const float& fov,const float& zoom) noexcept
	    : fov(fov),zoom(zoom),screen_width(w),screen_height(h),enable_ortho(false)
	{
	}

	void Camera::setFov(const float& val) noexcept
	{
	    if(val < 1.0f) {
			spdlog::warn("camera's fov can't be less than 1.0f (trying to set as {}), forcing to 1.0f",val);
	        fov = 1.0f;
	    }
	    else
	        fov = val;
	}

	glm::mat4 Camera::getMatrix() const noexcept
	{
		const auto& up = rotatable_point.up;
		const auto& quat = rotatable_point.orientation;

	    glm::vec3 glm_position(rotatable_point.x,rotatable_point.y,rotatable_point.z);
	    glm::vec3 glm_up(up[0],up[1],up[2]);
	    glm::quat glm_quat(quat[0],quat[1],quat[2],quat[3]);

	    glm::vec3 target = glm_position + glm::rotate(glm_quat, glm::vec3(0.0f, 0.0f, -1.0f));
	    
		if(enable_ortho)
			return glm::ortho(0.0f, (float)screen_width, 0.0f, (float)screen_height, 0.1f, 2000.0f) * glm::lookAt(glm_position, target, glm_up);
		else
			return glm::perspective(glm::radians(fov), (float)screen_width / screen_height, 0.1f, 2000.0f) * glm::lookAt(glm_position, target, glm_up);
	}
}