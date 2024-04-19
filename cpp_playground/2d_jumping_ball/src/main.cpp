#include <jumping_ball/gameobject.hpp>

using namespace jumping_ball::graphics;
using namespace jumping_ball::audio;
using namespace jumping_ball::physics;
using namespace jumping_ball::gameobject;

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

struct Ball : public GameObject {
	const float radius = 50.0f;

	Ball(std::shared_ptr<RenPipe> ren_pipe, std::unique_ptr<RigidBody> rigid_body) noexcept
		: GameObject(ren_pipe, std::move(rigid_body))
	{
		this->rigid_body->mass = 0.5f;
	}
};

static std::shared_ptr<std::vector<float>> ball_vertices = std::make_shared<std::vector<float>>();
static std::vector<unsigned int> ball_indices;

void genBallVerticesAndIndices() {
	const int segments = 360;
	ball_vertices->resize(segments * 3);
	ball_indices.resize(segments);

	for (int i = 0; i < segments; ++i) {
		float theta = 2.0f * 3.1415926f * float(i) / float(segments);
		float x = cosf(theta);
		float y = sinf(theta);

		ball_vertices->at(i * 3) = x;
		ball_vertices->at(i * 3 + 1) = y;
		ball_vertices->at(i * 3 + 2) = 0.0f; // z坐标，对于2D图形，可以简单地将其设置为0

		ball_indices[i] = i; // 设置索引值
	}
}

static std::shared_ptr<RenPipe> ball_ren_pipe;
static std::vector<std::unique_ptr<Ball>> balls;

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
		glm::vec2 direction = glm::vec2(mouseX, mouseY) - glm::vec2(balls[0]->rigid_body->position);

		// 将向量归一化，得到单位向量
		//glm::vec2 unitDirection = glm::normalize(direction);

		spdlog::debug("{},{}", direction.x, direction.y);

		balls[0]->rigid_body->velocity = glm::vec3(direction / 400.0f * static_cast<float>(delta_time), 0.0f);
	}
}

static constexpr float friction = 0.5f;
static constexpr float box_start_x = -800.0f;
static constexpr float box_start_y = -800.0f;
static constexpr float boxWidth = 800.0f;
static constexpr float boxHeight = 800.0f;

void try_play_ball_hit_sound(Ball& ball) noexcept {
	if (ball.rigid_body->velocity.x * ball.rigid_body->velocity.x + ball.rigid_body->velocity.y * ball.rigid_body->velocity.y >= 1.0f) {
		alSourcePlay(ball.audio_pipe.al_sources[0]);
	}
}

void check_hitbox_border(Ball& ball) noexcept {
	// check for collision with the box boundaries
	if (ball.rigid_body->position.x - ball.radius < box_start_x) {
		try_play_ball_hit_sound(ball);
		ball.rigid_body->position.x = ball.radius + box_start_x;
		ball.rigid_body->velocity.x = -ball.rigid_body->velocity.x * (1 - friction);
	}
	else if (ball.rigid_body->position.x + ball.radius > boxWidth) {
		try_play_ball_hit_sound(ball);
		ball.rigid_body->position.x = boxWidth - ball.radius;
		ball.rigid_body->velocity.x = -ball.rigid_body->velocity.x * (1 - friction);
	}

	if (ball.rigid_body->position.y - ball.radius < box_start_x) {
		try_play_ball_hit_sound(ball);
		ball.rigid_body->position.y = ball.radius + box_start_x;
		ball.rigid_body->velocity.y = -ball.rigid_body->velocity.y * (1 - friction);
	}
	else if (ball.rigid_body->position.y + ball.radius > boxHeight) {
		try_play_ball_hit_sound(ball);
		ball.rigid_body->position.y = boxHeight - ball.radius;
		ball.rigid_body->velocity.y = -ball.rigid_body->velocity.y * (1 - friction);
	}
}

void processTick() noexcept {
    for(auto& ball : balls) {
		ball->rigid_body->applyForce(glm::vec3(0.0f, -0.0025f, 0.0f), ball->rigid_body->position);
		ball->update(delta_time);
    }

    // 屎一样的碰撞检测以及响应
    for (size_t i = 0; i < balls.size(); ++i) {
		check_hitbox_border(*balls[i]);
        for (size_t j = i + 1; j < balls.size(); ++j) {
            //check_ball_collision(*balls[i], *balls[j]);
			for (auto& volume_a : balls[i]->rigid_body->bounding_volumes) {
				for (auto& volume_b : balls[j]->rigid_body->bounding_volumes) {
					if (volume_a->isIntersecting(*volume_b)) {
						glm::vec2 diff = glm::vec2(balls[i]->rigid_body->position - balls[j]->rigid_body->position);
						float dist = glm::length(diff);

						try_play_ball_hit_sound(*balls[i]);
						try_play_ball_hit_sound(*balls[j]);
						glm::vec2 norm = glm::normalize(diff);
						glm::vec2 relativeVelocity = glm::vec2(balls[i]->rigid_body->velocity - balls[j]->rigid_body->velocity);
						float speed = glm::dot(relativeVelocity, norm);

						if (speed < 0.0f) {
							float impulse = (1.0f + (1 - friction)) * speed / (1 / balls[i]->radius + 1 / balls[j]->radius);
							glm::vec2 impulseVec = impulse * norm;

							balls[i]->rigid_body->velocity -= glm::vec3(impulseVec / balls[i]->radius, 0.0f);
							balls[j]->rigid_body->velocity += glm::vec3(impulseVec / balls[j]->radius, 0.0f);

							// adjust positions to prevent overlap
							float overlap = 0.5f * (dist - balls[i]->radius - balls[j]->radius);
							balls[i]->rigid_body->position -= glm::vec3(overlap * norm, 0.0f);
							balls[j]->rigid_body->position += glm::vec3(overlap * norm, 0.0f);
						}
					}
				}
			}
        }
    }
}

void draw() noexcept {
	glClearColor(0.05f, 0.07f, 0.09f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw all ball(s)
	for (auto& ball : balls)
		ball->ren_pipe->draw(ball->rigid_body->position, 50.0f);
}

void mainLoop() noexcept {
	genBallVerticesAndIndices();
	ball_ren_pipe = std::make_shared<RenPipe>(vshader_source, fshader_source, *ball_vertices, ball_indices);
	glCheckError();

	for (int i = 0; i < 50; i++) {
		auto ball = std::make_unique<Ball>(ball_ren_pipe, std::make_unique<RigidBody>(ball_vertices));
		ball->rigid_body->bounding_volumes.emplace_back(std::make_unique<BoundingSphere>(glm::vec3(0.0f), 50.0f));
		balls.emplace_back(std::move(ball));
	}

	for (auto& ball : balls)
		for (auto& source : ball->audio_pipe.al_sources)
			alSourcei(source, AL_BUFFER, al_buffer);

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
	//spdlog::set_level(spdlog::level::debug);
	initAudio();
	initGraphics();
	mainLoop();
	closeGraphics();
	closeAudio();
	return 0;
}