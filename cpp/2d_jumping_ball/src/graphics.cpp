#include <jumping_ball/graphics.hpp>

GLFWwindow* jumping_ball::graphics::window;

void jumping_ball::graphics::initGraphics() noexcept {
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

void jumping_ball::graphics::closeGraphics() noexcept {
	glfwTerminate();
}

jumping_ball::graphics::RenPipe::RenPipe(const std::string_view& vshader_source, const std::string_view& fshader_source,
	const std::vector<float>& vertices, const std::vector<unsigned int>& indices) noexcept {
	initialize(vshader_source, fshader_source);
	updateVertices(vertices, indices);
}

jumping_ball::graphics::RenPipe::~RenPipe() noexcept {
	uninitiaze();
}

void jumping_ball::graphics::RenPipe::draw(const glm::vec3& position, glm::quat orientation, float scale) noexcept {
	// 计算变换矩阵
	// 暂时直接将窗口大小视为固定的800x800
	float window_scaled_r = scale / 400;
	float window_scaled_x = position.x / 400;
	float window_scaled_y = position.y / 400;

	glm::mat4 transform_mat = glm::translate(glm::mat4(1.0f), glm::vec3(window_scaled_x, window_scaled_y, 0.0f));
	glm::mat4 scale_mat = glm::scale(glm::mat4(1.0f), glm::vec3(window_scaled_r, window_scaled_r, window_scaled_r));
	glm::mat4 rotate_mat = glm::mat4_cast(orientation);

	// 将变换矩阵传递给着色器
	glUseProgram(shader_id);
	glUniformMatrix4fv(glGetUniformLocation(shader_id, "transform_mat"), 1, GL_FALSE, glm::value_ptr(transform_mat));
	glUniformMatrix4fv(glGetUniformLocation(shader_id, "rotate_mat"), 1, GL_FALSE, glm::value_ptr(rotate_mat));
	glUniformMatrix4fv(glGetUniformLocation(shader_id, "scale_mat"), 1, GL_FALSE, glm::value_ptr(scale_mat));

	// 临时
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	//glLineWidth(5.0f);

	// 绘制圆形
	glBindVertexArray(vao_id);
	glDrawElements(GL_TRIANGLES, indices_cnt, GL_UNSIGNED_INT, 0);
	//glDrawElements(GL_TRIANGLE_FAN, indices_cnt, GL_UNSIGNED_INT, 0);
}

void jumping_ball::graphics::RenPipe::initialize(const std::string_view& vshader_source,
	const std::string_view& fshader_source) noexcept {
	// 初始化VAO, VBO, EBO
	glGenVertexArrays(1, &vao_id);
	glGenBuffers(1, &vbo_id);
	glGenBuffers(1, &ebo_id);
	
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
	glDeleteVertexArrays(1, &vao_id);
	glDeleteBuffers(1, &vbo_id);
	glDeleteBuffers(1, &ebo_id);
	glDeleteProgram(shader_id);
}

void jumping_ball::graphics::RenPipe::updateVertices(const std::vector<float>& vertices,const std::vector<unsigned int>& indices) noexcept {
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