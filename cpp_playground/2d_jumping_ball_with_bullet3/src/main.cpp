#include <jumping_ball/gameobject.hpp>

using namespace jumping_ball;

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

struct Ball : public gameobject::GameObject {
	const float radius = BALL_RADIUS;

	Ball(std::shared_ptr<graphics::RenPipe> ren_pipe) noexcept
		: GameObject(ren_pipe)
	{
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

static std::shared_ptr<graphics::RenPipe> ball_ren_pipe;
static std::vector<std::unique_ptr<Ball>> balls;

static double delta_time = 0.0;
static double current_time = 0.0;
static double last_time = 0.0;

void processInput() noexcept {
	if (glfwGetKey(graphics::window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(graphics::window, true);

	if (glfwGetMouseButton(graphics::window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
		double mouse_x, mouse_y;
		glfwGetCursorPos(graphics::window, &mouse_x, &mouse_y);

		mouse_x = mouse_x - 400.0f;
		mouse_y = 400.0f - mouse_y;

		for (auto& ball : balls) {
			// 对小球施加速度
			glm::vec3 ball_position = ball->getPosition();
			glm::vec3 mouse_position(mouse_x, mouse_y, 0.0f);

			// 计算鼠标和小球之间的距离
			glm::vec3 direction = mouse_position - ball_position;
			direction = glm::normalize(direction);

			// 设置速度
			constexpr float speed = 100.0f;
			glm::vec3 velocity = direction * speed;

			// 设置小球的线性速度
			// 在Bullet物理引擎中，当一个刚体完全静止时，它会被引擎自动设置为休眠状态，以节省计算资源。
			// 所以这里可能需要强行唤醒
			ball->body->activate(true);
			ball->body->setLinearVelocity(btVector3(velocity.x, velocity.y, velocity.z));
		}
	}
}

void try_play_ball_hit_sound(gameobject::GameObject& go) noexcept {
	auto vel = go.getVelocity();
	float speed = sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z);

	if (speed >= 30.0f) {
		ALint source_state;
		alGetSourcei(go.audio_pipe.al_sources[0], AL_SOURCE_STATE, &source_state);
		if (source_state != AL_PLAYING) {
			spdlog::debug("speed = {}", speed);
			go.audio_pipe.setPosition(go.getPosition());
			go.audio_pipe.setVelocity(go.getVelocity());
			alSourceStop(go.audio_pipe.al_sources[0]);
			alSourcePlay(go.audio_pipe.al_sources[0]);
		}
	}
}

void processTick() noexcept {
	physics::processStepSimulation(delta_time);
	for (auto& ball : balls)
		ball->checkCollision();
}

void draw() noexcept {
	glClearColor(0.05f, 0.07f, 0.09f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw all ball(s)
	for (auto& ball : balls)
		ball->ren_pipe->draw(ball->getPosition(), ball->getRotate(), 1.0f);
}

void mainLoop() noexcept {
	genSphereVerticesAndIndices();
	ball_ren_pipe = std::make_shared<graphics::RenPipe>(vshader_source, fshader_source, *sphere_vertices, sphere_indices);
	graphics::glCheckError();

	for (int i = 0; i < 40; i++) {
		auto ball = std::make_unique<Ball>(ball_ren_pipe);
		ball->setCollisionCallback(try_play_ball_hit_sound);
		balls.emplace_back(std::move(ball));
	}

	for (auto& ball : balls)
		for (auto& source : ball->audio_pipe.al_sources)
			alSourcei(source, AL_BUFFER, audio::al_buffer);

	//physics::createHullFront();
	//physics::createHullBack();
	physics::createHullLeft();
	physics::createHullRight();
	physics::createHullUp();
	physics::createHullDown();

	while (!glfwWindowShouldClose(graphics::window)) {
		// update delta_time
		current_time = glfwGetTime() * 1000;
		delta_time = current_time - last_time;
		last_time = current_time;

		processInput();
		processTick();
		draw();
		graphics::glCheckError();

		glfwSwapBuffers(graphics::window);
		glfwPollEvents();
	}
}

int main() noexcept {
	//spdlog::set_level(spdlog::level::debug);
	physics::initialize();
	audio::initAudio();
	graphics::initGraphics();
	mainLoop();
	graphics::closeGraphics();
	audio::closeAudio();
	physics::uninitialize();
	return 0;
}