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

struct BallWithSoundSource : public Ball {
	ALuint sound_source;

	BallWithSoundSource() noexcept {
		alGenSources(1, &sound_source);
		alSourcei(sound_source, AL_BUFFER, al_buffer);
		alSourcei(sound_source, AL_LOOPING, 0);
		alSourcei(sound_source, AL_GAIN, 200.0f);
		alSourcei(sound_source, AL_PITCH, 1.0f);

		alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED);
		alSourcef(sound_source, AL_ROLLOFF_FACTOR, 1.0f);
	}

	~BallWithSoundSource() noexcept {
		alDeleteSources(1, &sound_source);
	}

	void updateSoundSource() noexcept {
		alSource3f(sound_source, AL_POSITION, position.x, position.y, 0.0f);
		alSource3f(sound_source, AL_VELOCITY, velocity.x, velocity.y, 0.0f);
	}
};

static std::unique_ptr<BallRenObject> ball_ren_obj;
static std::vector<std::unique_ptr<BallWithSoundSource>> balls;

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
		glm::vec2 direction = glm::vec2(mouseX, mouseY) - balls[0]->position;

		// 将向量归一化，得到单位向量
		//glm::vec2 unitDirection = glm::normalize(direction);

		spdlog::debug("{},{}", direction.x, direction.y);

		balls[0]->velocity = direction / 400.0f * static_cast<float>(delta_time);
	}
}

static constexpr float friction = 0.5f;
static constexpr float box_start_x = -800.0f;
static constexpr float box_start_y = -800.0f;
static constexpr float boxWidth = 800.0f;
static constexpr float boxHeight = 800.0f;

void apply_gravity(Ball& ball) noexcept {
	ball.velocity.y -= 0.0075f * delta_time;
}

void try_play_ball_hit_sound(BallWithSoundSource& ball) noexcept {
	if (ball.velocity.x * ball.velocity.x + ball.velocity.y * ball.velocity.y >= 1.0f) {
		ball.updateSoundSource();
		alSourcePlay(ball.sound_source);
	}
}

void check_hitbox_border(BallWithSoundSource& ball) noexcept {
	// check for collision with the box boundaries
	if (ball.position.x - ball.radius < box_start_x) {
		try_play_ball_hit_sound(ball);
		ball.position.x = ball.radius + box_start_x;
		ball.velocity.x = -ball.velocity.x * (1 - friction);
	}
	else if (ball.position.x + ball.radius > boxWidth) {
		try_play_ball_hit_sound(ball);
		ball.position.x = boxWidth - ball.radius;
		ball.velocity.x = -ball.velocity.x * (1 - friction);
	}

	if (ball.position.y - ball.radius < box_start_x) {
		try_play_ball_hit_sound(ball);
		ball.position.y = ball.radius + box_start_x;
		ball.velocity.y = -ball.velocity.y * (1 - friction);
	}
	else if (ball.position.y + ball.radius > boxHeight) {
		try_play_ball_hit_sound(ball);
		ball.position.y = boxHeight - ball.radius;
		ball.velocity.y = -ball.velocity.y * (1 - friction);
	}
}

void update_velocity_and_position(Ball& ball) noexcept {
	ball.position += ball.velocity * static_cast<float>(delta_time);
}

void check_ball_collision(BallWithSoundSource& ball1, BallWithSoundSource& ball2) noexcept {
    glm::vec2 diff = ball1.position - ball2.position;
    float dist = glm::length(diff);
    if (dist < ball1.radius + ball2.radius) {
		try_play_ball_hit_sound(ball1);
		try_play_ball_hit_sound(ball2);
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
    for(auto& ball : balls) {
        apply_gravity(*ball);
        check_hitbox_border(*ball);
        update_velocity_and_position(*ball);
    }

    // check for collision between balls
    for (size_t i = 0; i < balls.size(); ++i) {
        for (size_t j = i + 1; j < balls.size(); ++j) {
            check_ball_collision(*balls[i], *balls[j]);
        }
    }
}

void draw() noexcept {
	glClearColor(0.05f, 0.07f, 0.09f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// draw all ball(s)
	for(auto& ball : balls)
		ball_ren_obj->draw(*ball);
}

void mainLoop() noexcept {
	for(int i = 0;i < 50;i++)
		balls.emplace_back(std::make_unique<BallWithSoundSource>());
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