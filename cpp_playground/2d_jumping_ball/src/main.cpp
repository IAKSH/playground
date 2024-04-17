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
static std::vector<Ball> balls;

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
		// 暂时只有第一个小球能被拖动
		glm::vec2 direction = glm::vec2(mouseX, mouseY) - balls[0].position;

		// 将向量归一化，得到单位向量
		//glm::vec2 unitDirection = glm::normalize(direction);

		spdlog::debug("{},{}", direction.x, direction.y);

		balls[0].velocity = direction / 400.0f * static_cast<float>(delta_time);
	}
}

static constexpr float friction = 0.25f;
static constexpr float box_start_x = -800.0f;
static constexpr float box_start_y = -800.0f;
static constexpr float boxWidth = 800.0f;
static constexpr float boxHeight = 800.0f;

void apply_gravity(Ball& ball) noexcept {
	ball.velocity.y -= 0.0075f * delta_time;
}

void check_hitbox_border(Ball& ball) noexcept {
	// check for collision with the box boundaries
	if (ball.position.x - ball.radius < box_start_x) {
		alSourcePlay(al_source);
		ball.position.x = ball.radius + box_start_x;
		ball.velocity.x = -ball.velocity.x * (1 - friction);
	}
	else if (ball.position.x + ball.radius > boxWidth) {
		alSourcePlay(al_source);
		ball.position.x = boxWidth - ball.radius;
		ball.velocity.x = -ball.velocity.x * (1 - friction);
	}

	if (ball.position.y - ball.radius < box_start_x) {
		alSourcePlay(al_source);
		ball.position.y = ball.radius + box_start_x;
		ball.velocity.y = -ball.velocity.y * (1 - friction);
	}
	else if (ball.position.y + ball.radius > boxHeight) {
		alSourcePlay(al_source);
		ball.position.y = boxHeight - ball.radius;
		ball.velocity.y = -ball.velocity.y * (1 - friction);
	}
}

void update_velocity_and_position(Ball& ball) noexcept {
	ball.position += ball.velocity * static_cast<float>(delta_time);
}

void check_ball_collision(Ball& ball1, Ball& ball2) noexcept {
    glm::vec2 diff = ball1.position - ball2.position;
    float dist = glm::length(diff);
    if (dist < ball1.radius + ball2.radius) {
        //alSourcePlay(al_source);
        glm::vec2 norm = glm::normalize(diff);
        glm::vec2 relativeVelocity = ball1.velocity - ball2.velocity;
        float speed = glm::dot(relativeVelocity, norm);

        if (speed < 0.0f) {
            float impulse = (1.0f + (1 - friction)) * speed / (1 / ball1.radius + 1 / ball2.radius);
            glm::vec2 impulseVec = impulse * norm;

            ball1.velocity -= impulseVec / ball1.radius;
            ball2.velocity += impulseVec / ball2.radius;

			// adjust positions to prevent overlap
			float overlap = 0.5f * (dist - ball1.radius - ball2.radius);
			ball1.position -= overlap * norm;
			ball2.position += overlap * norm;
        }
    }
}

void processTick() noexcept {
    // apply gravity
    for(auto& ball : balls) {
        apply_gravity(ball);
        check_hitbox_border(ball);
        update_velocity_and_position(ball);
    }

    // check for collision between balls
    for (size_t i = 0; i < balls.size(); ++i) {
        for (size_t j = i + 1; j < balls.size(); ++j) {
            check_ball_collision(balls[i], balls[j]);
        }
    }

    // update audio source position
    // TODO: 每个ball一个al_source
    alSource3f(al_source,AL_POSITION,balls[0].position.x,balls[0].position.y,0.0f);
}

void draw() noexcept {
	glClearColor(0.05f, 0.07f, 0.09f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw all ball(s)
	for(auto& ball : balls)
		ball_ren_obj->draw(ball);
}

void mainLoop() noexcept {
	for(int i = 0;i < 10;i++)
		balls.emplace_back(Ball());
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