#include <jumping_ball/graphics.hpp>
#include <jumping_ball/audio.hpp>

using namespace jumping_ball::graphics;
using namespace jumping_ball::audio;
using namespace jumping_ball::physics;

static const std::string vshader_source = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 transform;

void main()
{
    gl_Position = transform * vec4(aPos, 1.0);
}
)";


static const std::string fshader_source = R"(
#version 430 core
out vec4 FragColor;

void main()
{
    FragColor = vec4(1.0f, 0.8f, 0.8f, 1.0f);
}
)";

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
	// apply gravity
	ball->velocity.y -= 0.0075f * delta_time;

	constexpr float friction = 0.25f;
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
	ball_ren_obj = std::make_unique<BallRenObject>(vshader_source, fshader_source);

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