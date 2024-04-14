#include <iostream>
#include <vector>
#include <AL/al.h>
#include <AL/alc.h>
#include <mpg123.h>
#include <spdlog/spdlog.h>

#define CHECK_OPENAL_ERRORS() check_al_errors(__FILE__, __LINE__)

void check_al_errors(const std::string& filename, const std::uint_fast32_t line) {
    ALenum error = alGetError();
    if (error != AL_NO_ERROR) {
        spdlog::error("OPENAL ERROR on line {} in {}: {}", line, filename, error);
    }
}

const std::string MP3_FILE_PATH = "a.mp3";

int main() {
    //spdlog::set_level(spdlog::level::debug);

    mpg123_handle* mh;
    unsigned char* buffer;
    size_t buffer_size;
    size_t done;
    int err;

    int channels, encoding;
    long rate;

    spdlog::info("initializing mpg123");

    // 初始化mpg123
    mpg123_init();
    mh = mpg123_new(NULL, &err);
    buffer_size = mpg123_outblock(mh);
    buffer = new unsigned char[buffer_size];

    spdlog::info("mpg123 ok");
    spdlog::info("openning mp3 file \"{}\"", MP3_FILE_PATH);

    // 打开mp3文件
    if (mpg123_open(mh, MP3_FILE_PATH.c_str()) != MPG123_OK) {
        spdlog::critical("can't open file \"{}\"", MP3_FILE_PATH);
        return -1;
    }

    spdlog::info("file loaded");
    spdlog::info("getting file format");

    // 获取音频信息
    if (mpg123_getformat(mh, &rate, &channels, &encoding) != MPG123_OK) {
        spdlog::critical("error getting format from \"{}\"", MP3_FILE_PATH);
        return -1;
    }

    // 如果是立体声，转换为单声道，可能造成音质损失
    // 因为OpenAL不能对立体声进行任何混音，只支持对单通道16位/8位的合成
    bool stereo2mono = false;
    if (channels != MPG123_MONO || encoding != MPG123_ENC_SIGNED_16) {
        spdlog::warn("force to mono, may cause some problem");
        mpg123_format_none(mh);
        mpg123_format(mh, rate, MPG123_MONO, MPG123_ENC_SIGNED_16);
        stereo2mono = true;
    }

    spdlog::info("file format ok");
    spdlog::info("initializing OpenAL");

    // 初始化OpenAL
    ALCdevice* device = alcOpenDevice(NULL);
    ALCcontext* context = alcCreateContext(device, NULL);
    alcMakeContextCurrent(context);

    ALuint source;
    alGenSources(1, &source);

    ALuint buffer_id;
    alGenBuffers(1, &buffer_id);

    // 设置监听者的位置
    ALfloat ListenerPos[] = { 0.0, 0.0, 0.0 };
    alListenerfv(AL_POSITION, ListenerPos);

    // 设置监听者的速度
    ALfloat ListenerVel[] = { 0.0, 0.0, 0.0 };
    alListenerfv(AL_VELOCITY, ListenerVel);

    // 设置监听者的方向
    ALfloat ListenerOri[] = { 0.0, 0.0, -1.0,  0.0, 1.0, 0.0 };
    alListenerfv(AL_ORIENTATION, ListenerOri);

    // 设置衰减
    alDistanceModel(AL_INVERSE_DISTANCE_CLAMPED);
    alSourcef(source, AL_ROLLOFF_FACTOR, 1.0);

    spdlog::info("OpenAL ok");
    spdlog::info("loading PCM");

    std::vector<ALshort> pcm;
    // 读取PCM数据
    while (mpg123_read(mh, buffer, buffer_size, &done) == MPG123_OK) {
        for (int i = 0; i < done / 2; i++) {
            pcm.push_back(reinterpret_cast<ALshort*>(buffer)[i]);
        }
    }

    // 将PCM数据加载到OpenAL缓冲区
    alBufferData(buffer_id, AL_FORMAT_MONO16, pcm.data(), pcm.size() * sizeof(ALshort), stereo2mono ? rate * 2 : rate);
    alSourcei(source, AL_BUFFER, buffer_id);
    alSourcei(source, AL_LOOPING, 1);
    alSourcef(source, AL_GAIN, 100.0f);
    alSourcef(source, AL_PITCH, 1.0f);
    CHECK_OPENAL_ERRORS();

    spdlog::info("PCM ready");
    spdlog::info("playing...");

    // 播放音频
    alSourcePlay(source);
    CHECK_OPENAL_ERRORS();

    bool should_exit = false;
    std::thread([&]() {
        // 按回车退出播放
        getchar();
        should_exit = true;
        }).detach();

    float pos[3]{ 0.0f,0.0f,0.0f };
    float i = 0.0f;
    while (!should_exit) {
        i += 0.0075f;
        pos[0] = sin(i) * 150;
        pos[1] = cos(i) * 150;
        spdlog::debug("pos[0] = {}\tpos[1] = {}\tpos[2] = {}", pos[0], pos[1], pos[2]);
        alSourcefv(source, AL_POSITION,pos);
        std::this_thread::sleep_for(std::chrono::milliseconds(32));
    }

    spdlog::info("finished, closing...");

    // 清理资源
    alDeleteSources(1, &source);
    alDeleteBuffers(1, &buffer_id);
    alcMakeContextCurrent(NULL);
    alcDestroyContext(context);
    alcCloseDevice(device);

    delete[] buffer;
    mpg123_close(mh);
    mpg123_delete(mh);
    mpg123_exit();

    spdlog::info("done");

    return 0;
}