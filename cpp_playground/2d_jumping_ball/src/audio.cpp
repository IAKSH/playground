#include <jumping_ball/audio.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <array>

ALCdevice* jumping_ball::audio::al_device;
ALCcontext* jumping_ball::audio::al_context;

ALuint jumping_ball::audio::al_buffer;

void jumping_ball::audio::initAudio() noexcept {
	mpg123_handle* mh;
	unsigned char* buffer;
	size_t buffer_size;
	size_t done;
	int err;
	int channels, encoding;
	long rate;
	int bits;
	bool stereo2mono = false;

	mpg123_init();
	mh = mpg123_new(nullptr, &err);
	buffer_size = mpg123_outblock(mh);
	buffer = new unsigned char[buffer_size];

	if (mpg123_open(mh, "hit.mp3") != MPG123_OK) {
		spdlog::critical("can't open \"hit.mp3\"");
		std::terminate();
	}

	if (mpg123_getformat(mh, &rate, &channels, &encoding) != MPG123_OK) {
		spdlog::critical("error getting format from \"hit.mp3\"");
		std::terminate();
	}

	switch (encoding)
	{
	case MPG123_ENC_SIGNED_16:
		bits = 16;
		break;
	case MPG123_ENC_SIGNED_24:
		bits = 24;
		break;
	case MPG123_ENC_SIGNED_32:
		bits = 32;
		break;
	default:
		bits = -1;
		break;
	}

	if (channels != MPG123_MONO || encoding != MPG123_ENC_SIGNED_16) {
		spdlog::warn("force to mono, may cause some problem");
		mpg123_format_none(mh);
		mpg123_format(mh, rate, MPG123_MONO, MPG123_ENC_SIGNED_16);
		stereo2mono = true;
	}

	al_device = alcOpenDevice(nullptr);
	al_context = alcCreateContext(al_device, nullptr);
	alcMakeContextCurrent(al_context);

	alGenBuffers(1, &al_buffer);

	ALfloat listener_pos[]{ 0.0f,0.0f,0.0f };
	alListenerfv(AL_POSITION, listener_pos);

	ALfloat listener_vec[]{ 0.0f,0.0f,0.0f };
	alListenerfv(AL_VELOCITY, listener_vec);

	ALfloat listener_dir[]{ 0.0f,0.0f,-1.0f,0.0f,1.0f,0.0f };
	alListenerfv(AL_ORIENTATION, listener_dir);

	std::vector<ALshort> pcm;
	while (mpg123_read(mh, buffer, buffer_size, &done) == MPG123_OK) {
		for (int i = 0; i < done / 2; i++) {
			pcm.emplace_back(reinterpret_cast<ALshort*>(buffer)[i]);
		}
	}

	alBufferData(al_buffer, AL_FORMAT_MONO16, pcm.data(), pcm.size() * sizeof(ALshort), stereo2mono ? rate * 2 : rate);

	delete[] buffer;
	mpg123_close(mh);
	mpg123_delete(mh);
	mpg123_exit();
}

void jumping_ball::audio::closeAudio() noexcept {
	alDeleteBuffers(1, &al_buffer);
	alcMakeContextCurrent(nullptr);
	alcDestroyContext(al_context);
	alcCloseDevice(al_device);
};

jumping_ball::audio::AudioPipe::AudioPipe(int source_num = 1) noexcept {
	addSource(source_num);
}

jumping_ball::audio::AudioPipe::~AudioPipe() noexcept {
	destroyAllALSources();
}

void jumping_ball::audio::AudioPipe::destroyAllALSources() noexcept {
	alDeleteSources(al_sources.size(), al_sources.data());
}

void jumping_ball::audio::AudioPipe::addSource(int source_num = 1) noexcept {
	al_sources.resize(al_sources.size() + source_num);
	alGenSources(source_num, al_sources.data() + sizeof(ALuint) * source_num);
}

void jumping_ball::audio::AudioPipe::setPosition(const glm::vec3& position) noexcept {
	for (auto& source : al_sources)
		alSourcefv(source, AL_POSITION, glm::value_ptr(position));
}

void jumping_ball::audio::AudioPipe::setVelocity(const glm::vec3& velocity) noexcept {
	for (auto& source : al_sources)
		alSourcefv(source, AL_VELOCITY, glm::value_ptr(velocity));
}

void jumping_ball::audio::AudioPipe::setOrientation(const glm::vec3& at, const glm::vec3& up) noexcept {
	std::array<ALfloat, 6> values{ at.x,at.y,at.z,up.x,up.y,up.z };
	for (auto& source : al_sources)
		alSourcefv(source, AL_ORIENTATION, values.data());
}