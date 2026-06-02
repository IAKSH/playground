#pragma once

#include <alsa/asoundlib.h>
#include <concepts>
#include <vector>

template<typename T>
concept __Player =
    std::constructible_from<T, unsigned int> &&
    requires(T t, const float* data, size_t len) {
        { t.prepare() } -> std::same_as<void>;
        { t.play(data, len) } -> std::same_as<void>;
        { t.drain() } -> std::same_as<void>;
    };

class ALSAPlayer {
public:
    ALSAPlayer(ALSAPlayer&) = delete;

    ALSAPlayer(unsigned int channels = 1);
    ~ALSAPlayer();

    // 0.0 ~ 2.0
    void set_volume(float v);
    void set_speed(float s);
    void prepare();
    void play(const float* data, size_t len);
    void drain();
    bool finished();

private:
    void setup_alsa_device();
    void delete_alsa_device();
    void resample(const float* data, size_t len);

    float volume;
    float speed;
    snd_pcm_t *pcm_handle;
    const unsigned int channels;
    std::vector<float> buffer;
};

static_assert(__Player<ALSAPlayer>);

using Player = ALSAPlayer;
