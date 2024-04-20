#include <jumping_ball/gameobject.hpp>

using namespace jumping_ball::graphics;
using namespace jumping_ball::audio;
using namespace jumping_ball::physics;
using namespace jumping_ball::gameobject;

static const std::string vshader_source = R"(
#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 transform_mat;
uniform mat4 rotate_mat;
uniform mat4 scale_mat;

void main()
{
    gl_Position =  transform_mat * scale_mat * rotate_mat * vec4(aPos, 1.0);
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

static constexpr float BALL_RADIUS = 25.0f;

struct Ball : public GameObject {
	const float radius = BALL_RADIUS;

	Ball(std::shared_ptr<RenPipe> ren_pipe, std::unique_ptr<RigidBody> rigid_body) noexcept
		: GameObject(ren_pipe, std::move(rigid_body))
	{
		this->rigid_body->mass = 1.0f;
	}
};

static std::shared_ptr<std::vector<float>> sphere_vertices = std::make_shared<std::vector<float>>();
static std::vector<unsigned int> sphere_indices;

void genSphereVerticesAndIndices(int segments = 6, float radius = BALL_RADIUS) {
	const float PI = 3.1415926f;
	for (int i = 0; i <= segments; ++i) {
		for (int j = 0; j <= segments; ++j) {
			float x_segment = (float)j / (float)segments;
			float y_segment = (float)i / (float)segments;
			float x = radius * cosf(2.0f * PI * x_segment) * sinf(PI * y_segment);
			float y = radius * cosf(PI * y_segment);
			float z = radius * sinf(2.0f * PI * x_segment) * sinf(PI * y_segment);

			sphere_vertices->push_back(x);
			sphere_vertices->push_back(y);
			sphere_vertices->push_back(z);
		}
	}

	for (int i = 0; i < segments; ++i) {
		for (int j = 0; j < segments; ++j) {
			int start = (i * (segments + 1)) + j;
			sphere_indices.push_back(start);
			sphere_indices.push_back(start + 1);
			sphere_indices.push_back(start + segments + 1);
			sphere_indices.push_back(start + segments + 1);
			sphere_indices.push_back(start + 1);
			sphere_indices.push_back(start + segments + 2);
		}
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

		//spdlog::debug("{},{}", direction.x, direction.y);

		balls[0]->rigid_body->velocity = glm::vec3(direction / 400.0f * static_cast<float>(delta_time), 0.0f);
	}
}

void try_play_ball_hit_sound(Ball& ball) noexcept {
	if (ball.rigid_body->velocity.x * ball.rigid_body->velocity.x + ball.rigid_body->velocity.y * ball.rigid_body->velocity.y >= 1.0f) {
		alSourcePlay(ball.audio_pipe.al_sources[0]);
	}
}

void handleBoxCollision(Ball& ball, float friction, float restitution) {
	constexpr float box_start_x = -400.0f;
	constexpr float box_start_y = -400.0f;
	constexpr float boxWidth = 800.0f;
	constexpr float boxHeight = 800.0f;

	// 计算球心与矩形框各边的距离
	float leftDist = ball.rigid_body->position.x - ball.radius - box_start_x;
	float rightDist = box_start_x + boxWidth - (ball.rigid_body->position.x + ball.radius);
	float topDist = ball.rigid_body->position.y - ball.radius - box_start_y;
	float bottomDist = box_start_y + boxHeight - (ball.rigid_body->position.y + ball.radius);

	// 检查是否发生碰撞，并计算碰撞后的速度
	if (leftDist < 0) {
		try_play_ball_hit_sound(ball);
		ball.rigid_body->velocity.x = restitution * abs(ball.rigid_body->velocity.x);
		ball.rigid_body->angular_velocity.y = friction * ball.rigid_body->velocity.x / ball.radius;
		ball.rigid_body->position.x = box_start_x + ball.radius; // 将球移动到矩形框内部的一个安全位置
	}
	else if (rightDist < 0) {
		try_play_ball_hit_sound(ball);
		ball.rigid_body->velocity.x = -restitution * abs(ball.rigid_body->velocity.x);
		ball.rigid_body->angular_velocity.y = -friction * ball.rigid_body->velocity.x / ball.radius;
		ball.rigid_body->position.x = box_start_x + boxWidth - ball.radius; // 将球移动到矩形框内部的一个安全位置
	}

	if (topDist < 0) {
		try_play_ball_hit_sound(ball);
		ball.rigid_body->velocity.y = restitution * abs(ball.rigid_body->velocity.y);
		ball.rigid_body->angular_velocity.x = -friction * ball.rigid_body->velocity.y / ball.radius;
		ball.rigid_body->position.y = box_start_y + ball.radius; // 将球移动到矩形框内部的一个安全位置
	}
	else if (bottomDist < 0) {
		try_play_ball_hit_sound(ball);
		ball.rigid_body->velocity.y = -restitution * abs(ball.rigid_body->velocity.y);
		ball.rigid_body->angular_velocity.x = friction * ball.rigid_body->velocity.y / ball.radius;
		ball.rigid_body->position.y = box_start_y + boxHeight - ball.radius; // 将球移动到矩形框内部的一个安全位置
	}
}

void handleBallCollision(Ball& ball_a, Ball& ball_b, float friction, float restitution) {
	// 计算两球之间的距离和速度差
	glm::vec3 distance = ball_b.rigid_body->position - ball_a.rigid_body->position;
	glm::vec3 velocityDifference = ball_b.rigid_body->velocity - ball_a.rigid_body->velocity;

	// 计算碰撞的方向
	glm::vec3 collisionDirection = glm::normalize(distance);

	// 计算相对速度在碰撞方向上的分量
	float velocityAlongCollisionDirection = glm::dot(velocityDifference, collisionDirection);

	// 如果速度在碰撞方向上的分量小于0，说明两球正在远离，无需处理碰撞
	if (velocityAlongCollisionDirection < 0) {
		return;
	}

	// 计算两球的旋转能量
	float rotationalEnergyThis = glm::length(ball_a.rigid_body->angular_velocity) * ball_a.rigid_body->inertia_tensor[0][0] / 2.0f;
	float rotationalEnergyOther = glm::length(ball_b.rigid_body->angular_velocity) * ball_b.rigid_body->inertia_tensor[0][0] / 2.0f;

	// 计算两球的动能
	float kineticEnergyThis = ball_a.rigid_body->mass * glm::length(ball_a.rigid_body->velocity) * glm::length(ball_a.rigid_body->velocity) / 2.0f;
	float kineticEnergyOther = ball_b.rigid_body->mass * glm::length(ball_b.rigid_body->velocity) * glm::length(ball_b.rigid_body->velocity) / 2.0f;

	// 计算碰撞前后的能量差
	float energyBefore = kineticEnergyThis + kineticEnergyOther + rotationalEnergyThis + rotationalEnergyOther;
	float energyAfter = energyBefore * restitution;

	// 计算碰撞后的速度
	float speedAfter = sqrt(2.0f * energyAfter / (ball_a.rigid_body->mass + ball_b.rigid_body->mass));

	// 更新两球的速度和旋转速度
	ball_a.rigid_body->velocity = (1 - friction) * speedAfter * (-collisionDirection);
	ball_b.rigid_body->velocity = (1 - friction) * speedAfter * collisionDirection;

	// 更新两球的旋转速度
	ball_a.rigid_body->angular_velocity = glm::cross(collisionDirection, ball_a.rigid_body->velocity) / (ball_a.radius * ball_a.radius);
	ball_b.rigid_body->angular_velocity = glm::cross(collisionDirection, ball_b.rigid_body->velocity) / (ball_b.radius * ball_b.radius);
}


void processTick() noexcept {
	spdlog::debug("{}\t{}\t{}", balls[0]->rigid_body->angular_velocity.x, balls[0]->rigid_body->angular_velocity.y, balls[0]->rigid_body->angular_velocity.z);

	for (auto& ball : balls) {
		// 应用重力
		glm::vec3 gravityForce = glm::vec3(0.0f, -0.005f * ball->rigid_body->mass, 0.0f);
		ball->rigid_body->applyForce(gravityForce, ball->rigid_body->position);
		ball->update(delta_time);
	}

	// 史一样的碰撞检测和响应
	for (size_t i = 0; i < balls.size(); ++i) {
		handleBoxCollision(*balls[i], 0.5f, 0.8f);
		for (size_t j = i + 1; j < balls.size(); ++j) {
			for (auto& volume_a : balls[i]->rigid_body->bounding_volumes) {
				for (auto& volume_b : balls[j]->rigid_body->bounding_volumes) {
					if (volume_a->isIntersecting(*volume_b)) {
						glm::vec3 diff = balls[i]->rigid_body->position - balls[j]->rigid_body->position;
						float dist = glm::length(diff);

						try_play_ball_hit_sound(*balls[i]);
						try_play_ball_hit_sound(*balls[j]);

						glm::vec3 norm = glm::normalize(diff);
						glm::vec3 relativeVelocity = balls[i]->rigid_body->velocity - balls[j]->rigid_body->velocity;
						float speed = glm::dot(relativeVelocity, norm);

						if (speed < 0.0f) {
							//respondToCollision(*balls[i], *balls[j], 0.5f, 0.8f);
							handleBallCollision(*balls[i], *balls[j], 0.5f, 0.8f);
							// 调整位置以防止重叠
							float overlap = 0.5f * (dist - balls[i]->radius - balls[j]->radius);
							balls[i]->rigid_body->position -= overlap * norm;
							balls[j]->rigid_body->position += overlap * norm;
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
		ball->ren_pipe->draw(ball->rigid_body->position, ball->rigid_body->orientation, 1.0f);
}

void mainLoop() noexcept {
	genSphereVerticesAndIndices();
	ball_ren_pipe = std::make_shared<RenPipe>(vshader_source, fshader_source, *sphere_vertices, sphere_indices);
	glCheckError();

	for (int i = 0; i < 2; i++) {
		auto ball = std::make_unique<Ball>(ball_ren_pipe, std::make_unique<RigidBody>(sphere_vertices));
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