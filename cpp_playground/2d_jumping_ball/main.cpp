#include <spdlog/spdlog.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <AL/al.h>
#include <AL/alc.h>
#include <mpg123.h>

static GLFWwindow* window;

GLenum glCheckError_(const char* file, int line)
{
	GLenum errorCode;
	while ((errorCode = glGetError()) != GL_NO_ERROR)
	{
		std::string error;
		switch (errorCode)
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
	return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__) 

const std::string vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 transform;

void main()
{
    gl_Position = transform * vec4(aPos, 1.0);
}
)";


// 片段着色器
const std::string fragmentShaderSource = R"(
#version 430 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0f, 0.8f, 0.8f, 1.0f);
}
)";

struct Ball {
	const float radius = 50.0f;
	glm::vec2 position{ 0.0f,0.0f };
	glm::vec2 velocity{ 0.0f,0.0f };
};

class BallRenObject {
public:
	GLuint VAO, VBO, EBO;
	GLuint shaderProgram;

	BallRenObject(
		const std::string& vertexShader = vertexShaderSource,
		const std::string& fragmentShader = fragmentShaderSource) {
		// 初始化VAO, VBO, EBO
		glGenVertexArrays(1, &VAO);
		glGenBuffers(1, &VBO);
		glGenBuffers(1, &EBO);

		// 初始化shader
		shaderProgram = initShader(vertexShader, fragmentShader);

		// 装载2D圆形的顶点数据
		loadCircleVertices();
	}

	~BallRenObject() noexcept {
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteBuffers(1, &EBO);
		glDeleteProgram(shaderProgram);
	}

	void draw(const Ball& ball) {
		// 计算变换矩阵
		// 暂时直接将窗口大小视为固定的800x800
		float window_scaled_r = ball.radius / 800;
		float window_scaled_x = ball.position.x / 800;
		float window_scaled_y = ball.position.y / 800;

		glm::mat4 transform = glm::mat4(1.0f);
		transform = glm::translate(transform, glm::vec3(window_scaled_x, window_scaled_y, 0.0f))
			* glm::scale(glm::mat4(1.0f), glm::vec3(window_scaled_r, window_scaled_r, window_scaled_r));
			

		// 将变换矩阵传递给着色器
		glUseProgram(shaderProgram);
		GLint transformLoc = glGetUniformLocation(shaderProgram, "transform");
		glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(transform));

		// 绘制圆形
		glBindVertexArray(VAO);
		//glDrawElements(GL_TRIANGLES, 360, GL_UNSIGNED_INT, 0);
		glDrawElements(GL_TRIANGLE_FAN, 360, GL_UNSIGNED_INT, 0);
	}

private:
	GLuint initShader(const std::string& vertexShaderSource, const std::string& fragmentShaderSource) {
		// 创建顶点着色器
		GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
		const char* vertexShaderSourceCStr = vertexShaderSource.c_str();
		glShaderSource(vertexShader, 1, &vertexShaderSourceCStr, NULL);
		glCompileShader(vertexShader);

		// 创建片段着色器
		GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		const char* fragmentShaderSourceCStr = fragmentShaderSource.c_str();
		glShaderSource(fragmentShader, 1, &fragmentShaderSourceCStr, NULL);
		glCompileShader(fragmentShader);

		// 创建着色器程序
		GLuint shaderProgram = glCreateProgram();
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);

		// 删除着色器，它们已经链接到我们的程序中了
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		return shaderProgram;
	}

	void loadCircleVertices() {
		const int segments = 360; // 分段数，你可以根据需要调整
		float vertices[segments * 3]; // 每个顶点有x, y, z三个坐标
		GLuint indices[segments]; // 索引数组

		for (int i = 0; i < segments; ++i) {
			float theta = 2.0f * 3.1415926f * float(i) / float(segments); // 当前角度
			float x = cosf(theta); // x坐标
			float y = sinf(theta); // y坐标

			vertices[i * 3] = x;
			vertices[i * 3 + 1] = y;
			vertices[i * 3 + 2] = 0.0f; // z坐标，对于2D图形，我们可以简单地将其设置为0

			indices[i] = i; // 设置索引值
		}

		glBindVertexArray(VAO);

		glBindBuffer(GL_ARRAY_BUFFER, VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		glBindVertexArray(0);
	}
};

void initGraphics() noexcept {
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

void closeGraphics() noexcept {
	glfwTerminate();
}

static ALCdevice* al_device;
static ALCcontext* al_context;

static ALuint al_source;
static ALuint al_buffer;

void initAudio() noexcept {
	mpg123_handle* mh;
	unsigned char* buffer;
	size_t buffer_size;
	size_t done;
	int err;
	int channels,encoding;
	long rate;
	int bits;
	bool stereo2mono = false;

	mpg123_init();
	mh = mpg123_new(nullptr,&err);
	buffer_size = mpg123_outblock(mh);
	buffer = new unsigned char[buffer_size];

	if(mpg123_open(mh,"hit.mp3") != MPG123_OK) {
		spdlog::critical("can't open \"hit.mp3\"");
		std::terminate();
	}

	if(mpg123_getformat(mh,&rate,&channels,&encoding) != MPG123_OK) {
		spdlog::critical("error getting format from \"hit.mp3\"");
		std::terminate();
	}

	switch (encoding)
	{
	case MPG123_ENC_SIGNED_16:
		bits = 16;
		break;
	case MPG123_ENC_SIGNED_24:
		bits = 24;
		break;
	case MPG123_ENC_SIGNED_32:
		bits = 32;
		break;
	default:
		bits = -1;
		break;
	}

	if(channels != MPG123_MONO || encoding != MPG123_ENC_SIGNED_16) {
		spdlog::warn("force to mono, may cause some problem");
		mpg123_format_none(mh);
		mpg123_format(mh,rate,MPG123_MONO,MPG123_ENC_SIGNED_16);
		stereo2mono = true;
	}

	al_device = alcOpenDevice(nullptr);
	al_context = alcCreateContext(al_device,nullptr);
	alcMakeContextCurrent(al_context);

	alGenSources(1,&al_source);
	alGenBuffers(1,&al_buffer);

	ALfloat listener_pos[]{0.0f,0.0f,0.0f};
	alListenerfv(AL_POSITION,listener_pos);

	ALfloat listener_vec[]{0.0f,0.0f,0.0f};
	alListenerfv(AL_VELOCITY,listener_vec);

	ALfloat listener_dir[]{0.0f,0.0f,-1.0f,0.0f,1.0f,0.0f};
	alListenerfv(AL_ORIENTATION,listener_dir);

	alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED);
	alSourcef(al_source,AL_ROLLOFF_FACTOR,1.0f);

	std::vector<ALshort> pcm;
	while(mpg123_read(mh,buffer,buffer_size,&done) == MPG123_OK) {
		for(int i = 0;i < done / 2;i++) {
			pcm.emplace_back(reinterpret_cast<ALshort*>(buffer)[i]);
		}
	}

	alBufferData(al_buffer,AL_FORMAT_MONO16,pcm.data(),pcm.size() * sizeof(ALshort),stereo2mono ? rate * 2 : rate);
	alSourcei(al_source,AL_BUFFER,al_buffer);
	alSourcei(al_source,AL_LOOPING,0);
	alSourcei(al_source,AL_GAIN,200.0f);
	alSourcei(al_source,AL_PITCH,1.0f);

	delete[] buffer;
	mpg123_close(mh);
	mpg123_delete(mh);
	mpg123_exit();
}

void closeAudio() noexcept {
	alDeleteSources(1,&al_source);
	alDeleteBuffers(1,&al_buffer);
	alcMakeContextCurrent(nullptr);
	alcDestroyContext(al_context);
	alcCloseDevice(al_device);
}

static std::unique_ptr<BallRenObject> ball_ren_obj;
static std::unique_ptr<Ball> ball;

static double delta_time = 0.0;
static double current_time = 0.0;
static double last_time = 0.0;

void processInput() noexcept {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
		double mouseX, mouseY;
		// 获取鼠标的位置
		glfwGetCursorPos(window, &mouseX, &mouseY);

		// 转换鼠标坐标
		mouseX = mouseX - 400.0f;
		mouseY = 400.0f - mouseY; // 注意这里取了反

		// 计算鼠标和小球之间的向量
		glm::vec2 direction = glm::vec2(mouseX, mouseY) - ball->position;

		// 将向量归一化，得到单位向量
		//glm::vec2 unitDirection = glm::normalize(direction);

		spdlog::debug("{},{}", direction.x, direction.y);

		ball->velocity = direction / 400.0f * static_cast<float>(delta_time);
	}
}

void processTick() noexcept {
	// update velocity
	//ball->velocity.x = sin(glfwGetTime()) / 10.0f;
	//ball->velocity.y = cos(glfwGetTime()) / 10.0f;

	// apply gravity
	ball->velocity.y -= 0.0075f * delta_time;

	constexpr float friction = 0.5f;
	constexpr float box_start_x = -800.0f;
	constexpr float box_start_y = -800.0f;
	constexpr float boxWidth = 800.0f;
	constexpr float boxHeight = 800.0f;

	// check for collision with the box boundaries
	if (ball->position.x - ball->radius < box_start_x) {
		alSourcePlay(al_source);
		ball->position.x = ball->radius + box_start_x;
		ball->velocity.x = -ball->velocity.x * (1 - friction);
	}
	else if (ball->position.x + ball->radius > boxWidth) {
		alSourcePlay(al_source);
		ball->position.x = boxWidth - ball->radius;
		ball->velocity.x = -ball->velocity.x * (1 - friction);
	}

	if (ball->position.y - ball->radius < box_start_x) {
		alSourcePlay(al_source);
		ball->position.y = ball->radius + box_start_x;
		ball->velocity.y = -ball->velocity.y * (1 - friction);
	}
	else if (ball->position.y + ball->radius > boxHeight) {
		alSourcePlay(al_source);
		ball->position.y = boxHeight - ball->radius;
		ball->velocity.y = -ball->velocity.y * (1 - friction);
	}

	// update velocity & position
	ball->position += ball->velocity * static_cast<float>(delta_time);

	// update audio source position
	alSource3f(al_source,AL_POSITION,ball->position.x,ball->position.y,0.0f);
}

void draw() noexcept {
	glClearColor(0.05f, 0.07f, 0.09f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	ball_ren_obj->draw(*ball);
}

void mainLoop() noexcept {
	ball = std::make_unique<Ball>();
	ball_ren_obj = std::make_unique<BallRenObject>();

	while (!glfwWindowShouldClose(window)) {
		// update delta_time
		current_time = glfwGetTime() * 1000;
		delta_time = current_time - last_time;
		last_time = current_time;

		processInput();
		processTick();
		draw();
		glCheckError();
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

int main() noexcept {
	spdlog::set_level(spdlog::level::debug);
	initAudio();
	initGraphics();
	mainLoop();
	closeGraphics();
	closeAudio();
	return 0;
}