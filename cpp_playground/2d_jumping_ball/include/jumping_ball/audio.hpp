#pragma once

#include <AL/al.h>
#include <AL/alc.h>
#include <mpg123.h>
#include <spdlog/spdlog.h>
#include <glm/vec3.hpp>

namespace jumping_ball::audio {
	extern ALCdevice* al_device;
	extern ALCcontext* al_context;
	extern ALuint al_buffer;

	void initAudio() noexcept;
	void closeAudio() noexcept;

	class AudioPipe {
	public:
		std::vector <ALuint> al_sources;

		AudioPipe(int source_num) noexcept;
		~AudioPipe() noexcept;
		void setPosition(const glm::vec3& position) noexcept;
		void setVelocity(const glm::vec3& velocity) noexcept;
		void setOrientation(const glm::vec3& at,const glm::vec3& up) noexcept;
		void addSource(int source_num) noexcept;

	private:
		void destroyAllALSources() noexcept;
	};
}