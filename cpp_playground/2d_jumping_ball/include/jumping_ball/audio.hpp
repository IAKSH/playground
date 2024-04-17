#pragma once

#include <AL/al.h>
#include <AL/alc.h>
#include <mpg123.h>
#include <spdlog/spdlog.h>

namespace jumping_ball::audio {
	extern ALCdevice* al_device;
	extern ALCcontext* al_context;

	extern ALuint al_source;
	extern ALuint al_buffer;

	void initAudio() noexcept;
	void closeAudio() noexcept;
}