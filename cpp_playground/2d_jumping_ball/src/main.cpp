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

static std::shared_ptr<RenPipe> ball_ren_pipe;
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
		glm::vec2 direction = glm::vec2(mouseX, mouseY) - glm::vec2(balls[0]->position);

		// 将向量归一化，得到单位向量
		//glm::vec2 unitDirection = glm::normalize(direction);

		spdlog::debug("{},{}", direction.x, direction.y);

		balls[0]->velocity = glm::vec3(direction / 400.0f * static_cast<float>(delta_time), 0.0f);
	}
}

static constexpr float friction = 0.5f;
static constexpr float box_start_x = -800.0f;
static constexpr float box_start_y = -800.0f;
static constexpr float boxWidth = 800.0f;
static constexpr float boxHeight = 800.0f;

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

void check_ball_collision(BallWithSoundSource& ball1, BallWithSoundSource& ball2) noexcept {
    glm::vec2 diff = glm::vec2(ball1.position - ball2.position);
    float dist = glm::length(diff);
    if (dist < ball1.radius + ball2.radius) {
		try_play_ball_hit_sound(ball1);
		try_play_ball_hit_sound(ball2);
        glm::vec2 norm = glm::normalize(diff);
        glm::vec2 relativeVelocity = glm::vec2(ball1.velocity - ball2.velocity);
        float speed = glm::dot(relativeVelocity, norm);

        if (speed < 0.0f) {
            float impulse = (1.0f + (1 - friction)) * speed / (1 / ball1.radius + 1 / ball2.radius);
            glm::vec2 impulseVec = impulse * norm;

			ball1.velocity -= glm::vec3(impulseVec / ball1.radius, 0.0f);
			ball2.velocity += glm::vec3(impulseVec / ball2.radius, 0.0f);

			// adjust positions to prevent overlap
			float overlap = 0.5f * (dist - ball1.radius - ball2.radius);
			ball1.position -= glm::vec3(overlap * norm, 0.0f);
			ball2.position += glm::vec3(overlap * norm, 0.0f);
        }
    }
}

static std::vector<float> ball_vertices;
static std::vector<unsigned int> ball_indices;

void genBallVerticesAndIndices() {
	const int segments = 360;
	ball_vertices.resize(segments * 3);
	ball_indices.resize(segments);

	for (int i = 0; i < segments; ++i) {
		float theta = 2.0f * 3.1415926f * float(i) / float(segments);
		float x = cosf(theta);
		float y = sinf(theta);

		ball_vertices[i * 3] = x;
		ball_vertices[i * 3 + 1] = y;
		ball_vertices[i * 3 + 2] = 0.0f; // z坐标，对于2D图形，可以简单地将其设置为0

		ball_indices[i] = i; // 设置索引值
	}

	/*
	glBindVertexArray(vao_id);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_id);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_id);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	*/
}

void processTick() noexcept {
    for(auto& ball : balls) {
		ball->applyForce(glm::vec3(0.0f, -0.0025f, 0.0f), ball->position);
        check_hitbox_border(*ball);
		ball->update(delta_time);
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
		ball_ren_pipe->draw(ball->position,ball->radius);
}

void mainLoop() noexcept {
	genBallVerticesAndIndices();
	ball_ren_pipe = std::make_shared<RenPipe>(vshader_source, fshader_source, ball_vertices, ball_indices);

	for (int i = 0; i < 50; i++)
		balls.emplace_back(std::make_unique<BallWithSoundSource>());

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