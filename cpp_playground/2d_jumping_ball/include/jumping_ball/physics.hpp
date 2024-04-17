#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace jumping_ball::physics {
	struct Ball {
		const float radius = 50.0f;
		glm::vec2 position{ 0.0f,0.0f };
		glm::vec2 velocity{ 0.0f,0.0f };
	};
}