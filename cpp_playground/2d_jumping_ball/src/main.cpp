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

void applyGravityAndUpdate(Ball& ball, float delta_time) {
	// 应用重力
	glm::vec3 gravityForce = glm::vec3(0.0f, -0.005f * ball.rigid_body->mass, 0.0f);
	ball.rigid_body->applyForce(gravityForce, ball.rigid_body->position);

	// 更新状态
	ball.update(delta_time);
}


void try_play_ball_hit_sound(Ball& ball) noexcept {
	float speed = sqrt(ball.rigid_body->velocity.x * ball.rigid_body->velocity.x
		+ ball.rigid_body->velocity.y * ball.rigid_body->velocity.y
		+ ball.rigid_body->velocity.z * ball.rigid_body->velocity.z);

	if (speed >= 0.5f) {
		//spdlog::debug("speed = {}", speed);
		alSourceStop(ball.audio_pipe.al_sources[0]);
		alSourcePlay(ball.audio_pipe.al_sources[0]);
	}
}

void handleBoxCollision(Ball& ball, float restitution, float friction) {
	glm::vec3 min(-400, -400, 0);
	glm::vec3 max(400, 400, 0);

	// 不检查Z轴
	for (int i = 0; i < 2; ++i) {
		if (ball.rigid_body->position[i] - ball.radius < min[i]) {
			glm::vec3 normal = glm::vec3(i == 0 ? 1 : 0, i == 1 ? 1 : 0, 0);
			glm::vec3 tangent = glm::cross(normal, ball.rigid_body->angular_velocity);
			glm::vec3 relative_velocity = ball.rigid_body->velocity - tangent;
			float velocity_along_normal = glm::dot(relative_velocity, normal);
			float impulse_scalar = -(1 + restitution) * velocity_along_normal / ball.rigid_body->mass;
			glm::vec3 impulse = impulse_scalar * normal;
			ball.rigid_body->velocity += impulse / ball.rigid_body->mass;
			ball.rigid_body->angular_velocity += friction * impulse / ball.radius;
			ball.rigid_body->position[i] = min[i] + ball.radius;  // 修正位置
			try_play_ball_hit_sound(ball);  // 播放音效
		}
		else if (ball.rigid_body->position[i] + ball.radius > max[i]) {
			glm::vec3 normal = glm::vec3(i == 0 ? -1 : 0, i == 1 ? -1 : 0, 0);
			glm::vec3 tangent = glm::cross(normal, ball.rigid_body->angular_velocity);
			glm::vec3 relative_velocity = ball.rigid_body->velocity - tangent;
			float velocity_along_normal = glm::dot(relative_velocity, normal);
			float impulse_scalar = -(1 + restitution) * velocity_along_normal / ball.rigid_body->mass;
			glm::vec3 impulse = impulse_scalar * normal;
			ball.rigid_body->velocity += impulse / ball.rigid_body->mass;
			ball.rigid_body->angular_velocity += friction * impulse / ball.radius;
			ball.rigid_body->position[i] = max[i] - ball.radius;  // 修正位置
			try_play_ball_hit_sound(ball);  // 播放音效
		}
	}
}

void checkAndHandleBallCollision(Ball& ball1, Ball& ball2, float restitution, float friction) {
	glm::vec3 diff = ball1.rigid_body->position - ball2.rigid_body->position;
	float dist = glm::length(diff);
	float radius_sum = ball1.radius + ball2.radius;

	if (dist <= 0.0f)
		return;

	if (dist < radius_sum) {
		glm::vec3 normal = glm::normalize(diff);
		glm::vec3 tangent1 = glm::cross(normal, ball1.rigid_body->angular_velocity);
		glm::vec3 tangent2 = glm::cross(normal, ball2.rigid_body->angular_velocity);
		glm::vec3 relative_velocity = ball1.rigid_body->velocity - ball2.rigid_body->velocity - tangent1 + tangent2;
		float velocity_along_normal = glm::dot(relative_velocity, normal);

		if (velocity_along_normal > 0) {
			return;
		}

		float impulse_scalar = -(1 + restitution) * velocity_along_normal;
		impulse_scalar /= ball1.rigid_body->mass + ball2.rigid_body->mass;

		glm::vec3 impulse = impulse_scalar * normal;

		ball1.rigid_body->velocity += impulse / ball1.rigid_body->mass;
		ball1.rigid_body->angular_velocity += friction * impulse / ball1.radius;
		ball2.rigid_body->velocity -= impulse / ball2.rigid_body->mass;
		ball2.rigid_body->angular_velocity -= friction * impulse / ball2.radius;

		// 修正位置
		float penetration = radius_sum - dist;
		glm::vec3 correction = (penetration / (ball1.rigid_body->mass + ball2.rigid_body->mass)) * normal;
		ball1.rigid_body->position += correction * ball1.rigid_body->mass;
		ball2.rigid_body->position -= correction * ball2.rigid_body->mass;
	}
}


void processTick() noexcept {
	spdlog::debug("{}\t{}\t{}", balls[0]->rigid_body->angular_velocity.x, balls[0]->rigid_body->angular_velocity.y, balls[0]->rigid_body->angular_velocity.z);

	for (auto& ball : balls) {
		applyGravityAndUpdate(*ball, delta_time);
	}

	// 碰撞检测然后响应
	for (size_t i = 0; i < balls.size(); ++i) {
		handleBoxCollision(*balls[i], 0.8f, 0.5f);
		for (size_t j = i + 1; j < balls.size(); ++j) {
			checkAndHandleBallCollision(*balls[i], *balls[j], 0.8f, 0.5f);
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
		ball->rigid_body->position.x = -400.0f + ball->radius * i;
		ball->rigid_body->position.y = -400.0f + ball->radius * i;
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
	spdlog::set_level(spdlog::level::debug);
	initAudio();
	initGraphics();
	mainLoop();
	closeGraphics();
	closeAudio();
	return 0;
}