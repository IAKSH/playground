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

jumping_ball::graphics::BallRenObject::BallRenObject(
	const std::string& vshader,
	const std::string& fshader) {
	// 初始化VAO, VBO, EBO
	glGenVertexArrays(1, &vao_id);
	glGenBuffers(1, &vbo_id);
	glGenBuffers(1, &ebo_id);

	// 初始化shader
	initShader(vshader, fshader);

	// 装载2D圆形的顶点数据
	loadCircleVertices();
}

jumping_ball::graphics::BallRenObject::~BallRenObject() noexcept {
	glDeleteVertexArrays(1, &vao_id);
	glDeleteBuffers(1, &vbo_id);
	glDeleteBuffers(1, &ebo_id);
	glDeleteProgram(shader_id);
}

void jumping_ball::graphics::BallRenObject::draw(const physics::Ball& ball) {
	// 计算变换矩阵
	// 暂时直接将窗口大小视为固定的800x800
	float window_scaled_r = ball.radius / 800;
	float window_scaled_x = ball.position.x / 800;
	float window_scaled_y = ball.position.y / 800;

	glm::mat4 transform = glm::mat4(1.0f);
	transform = glm::translate(transform, glm::vec3(window_scaled_x, window_scaled_y, 0.0f))
		* glm::scale(glm::mat4(1.0f), glm::vec3(window_scaled_r, window_scaled_r, window_scaled_r));


	// 将变换矩阵传递给着色器
	glUseProgram(shader_id);
	GLint transform_loc = glGetUniformLocation(shader_id, "transform");
	glUniformMatrix4fv(transform_loc, 1, GL_FALSE, glm::value_ptr(transform));

	// 绘制圆形
	glBindVertexArray(vao_id);
	//glDrawElements(GL_TRIANGLES, 360, GL_UNSIGNED_INT, 0);
	glDrawElements(GL_TRIANGLE_FAN, 360, GL_UNSIGNED_INT, 0);
}

void jumping_ball::graphics::BallRenObject::initShader(const std::string& vshader_source, const std::string& fshader_source) {
	// 创建顶点着色器
	GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
	const char* vshader_source_cstr = vshader_source.c_str();
	glShaderSource(vertex_shader, 1, &vshader_source_cstr, NULL);
	glCompileShader(vertex_shader);

	// 创建片段着色器
	GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
	const char* fshader_source_cstr = fshader_source.c_str();
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

void jumping_ball::graphics::BallRenObject::loadCircleVertices() {
	const int segments = 360; // 分段数
	float vertices[segments * 3]; // 每个顶点有x, y, z三个坐标
	GLuint indices[segments]; // 索引数组

	for (int i = 0; i < segments; ++i) {
		float theta = 2.0f * 3.1415926f * float(i) / float(segments); // 当前角度
		float x = cosf(theta); // x坐标
		float y = sinf(theta); // y坐标

		vertices[i * 3] = x;
		vertices[i * 3 + 1] = y;
		vertices[i * 3 + 2] = 0.0f; // z坐标，对于2D图形，可以简单地将其设置为0

		indices[i] = i; // 设置索引值
	}

	glBindVertexArray(vao_id);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_id);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}