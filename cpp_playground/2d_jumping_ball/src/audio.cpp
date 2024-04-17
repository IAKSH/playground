#include <jumping_ball/audio.hpp>

ALCdevice* jumping_ball::audio::al_device;
ALCcontext* jumping_ball::audio::al_context;

ALuint jumping_ball::audio::al_source;
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

	alGenSources(1, &al_source);
	alGenBuffers(1, &al_buffer);

	ALfloat listener_pos[]{ 0.0f,0.0f,0.0f };
	alListenerfv(AL_POSITION, listener_pos);

	ALfloat listener_vec[]{ 0.0f,0.0f,0.0f };
	alListenerfv(AL_VELOCITY, listener_vec);

	ALfloat listener_dir[]{ 0.0f,0.0f,-1.0f,0.0f,1.0f,0.0f };
	alListenerfv(AL_ORIENTATION, listener_dir);

	alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED);
	alSourcef(al_source, AL_ROLLOFF_FACTOR, 1.0f);

	std::vector<ALshort> pcm;
	while (mpg123_read(mh, buffer, buffer_size, &done) == MPG123_OK) {
		for (int i = 0; i < done / 2; i++) {
			pcm.emplace_back(reinterpret_cast<ALshort*>(buffer)[i]);
		}
	}

	alBufferData(al_buffer, AL_FORMAT_MONO16, pcm.data(), pcm.size() * sizeof(ALshort), stereo2mono ? rate * 2 : rate);
	alSourcei(al_source, AL_BUFFER, al_buffer);
	alSourcei(al_source, AL_LOOPING, 0);
	alSourcei(al_source, AL_GAIN, 200.0f);
	alSourcei(al_source, AL_PITCH, 1.0f);

	delete[] buffer;
	mpg123_close(mh);
	mpg123_delete(mh);
	mpg123_exit();
}

void jumping_ball::audio::closeAudio() noexcept {
	alDeleteSources(1, &al_source);
	alDeleteBuffers(1, &al_buffer);
	alcMakeContextCurrent(nullptr);
	alcDestroyContext(al_context);
	alcCloseDevice(al_device);
};