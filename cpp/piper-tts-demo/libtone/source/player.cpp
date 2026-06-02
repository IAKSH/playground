#include <tone/player.hpp>
#include <spdlog/spdlog.h>
#include <algorithm>

ALSAPlayer::ALSAPlayer(unsigned int channels)
    : channels{channels},volume{1.0f},speed{1.0f} {
    setup_alsa_device();
}

ALSAPlayer::~ALSAPlayer() {
    delete_alsa_device();
}

void ALSAPlayer::set_volume(float v) {
    volume = (v >= 0.01f ? v : 0.01f);
}

void ALSAPlayer::set_speed(float s) {
    speed = (s >= 0.01f ? s : 0.01f);
}

void ALSAPlayer::prepare() {
    snd_pcm_prepare(pcm_handle);
}

/*
void ALSAPlayer::play(const float* data, size_t len) {
    buffer.resize(len);
    for (size_t i = 0; i < len; ++i) {
        buffer[i] = data[i] * volume;
    }

    const float* pcm = buffer.data();

    size_t frames_total = len / channels;
    size_t frames_written = 0;
    // snd_pcm_writei 可能不会一次写完所有数据，需循环写入直到全部写完。
    while (frames_written < frames_total) {
        snd_pcm_sframes_t frames = snd_pcm_writei(
            pcm_handle,
            pcm + frames_written * channels,
            frames_total - frames_written
        );
        if (frames < 0) {
            frames = snd_pcm_recover(pcm_handle, frames, 0);
            if (frames < 0) {
                spdlog::error("ALSA write error: {}", snd_strerror(frames));
                break;
            }
            continue; // recover后重试
        }
        frames_written += frames;
    }
}
*/

void ALSAPlayer::play(const float* data, size_t len) {
    if (speed != 1.0f)
        // 应用speed和volume，保存到buffer
        resample(data, len);
    else {
        // 单独应用volume
        buffer.resize(len);
        for (size_t i = 0; i < len; ++i)
            buffer[i] = data[i] * volume;
    }

    const float* pcm = buffer.data();
    size_t frames_total = buffer.size() / channels;
    size_t frames_written = 0;

    while (frames_written < frames_total) {
        snd_pcm_sframes_t frames = snd_pcm_writei(
            pcm_handle,
            pcm + frames_written * channels,
            frames_total - frames_written
        );

        if (frames < 0) {
            frames = snd_pcm_recover(pcm_handle, frames, 0);
            if (frames < 0) {
                spdlog::error("ALSA write error: {}", snd_strerror(frames));
                break;
            }
            continue;
        }

        frames_written += frames;
    }
}

void ALSAPlayer::drain() {
    snd_pcm_drain(pcm_handle);
} 

bool ALSAPlayer::finished() {
    auto state = snd_pcm_state(pcm_handle);
    return (state == SND_PCM_STATE_PREPARED || state == SND_PCM_STATE_SETUP);
} 

void ALSAPlayer::setup_alsa_device() {
    snd_pcm_hw_params_t *params;
    unsigned int sample_rate = 22050;
    snd_pcm_format_t format = SND_PCM_FORMAT_FLOAT_LE; // Piper 输出 float32 小端序
    // 打开默认声卡
    if (snd_pcm_open(&pcm_handle, "default", SND_PCM_STREAM_PLAYBACK, 0) < 0) {
        spdlog::error("Error opening default PCM device");
        std::terminate();
    }
    snd_pcm_hw_params_malloc(&params);
    snd_pcm_hw_params_any(pcm_handle, params);
    snd_pcm_hw_params_set_access(pcm_handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm_handle, params, format);
    snd_pcm_hw_params_set_channels(pcm_handle, params, channels);
    snd_pcm_hw_params_set_rate_near(pcm_handle, params, &sample_rate, nullptr);
    snd_pcm_hw_params(pcm_handle, params);
    snd_pcm_hw_params_free(params);
    snd_pcm_prepare(pcm_handle);
}

void ALSAPlayer::delete_alsa_device() {
    if(pcm_handle) {
        snd_pcm_drain(pcm_handle);
        snd_pcm_close(pcm_handle);
    }   
}

// 线性插值
void ALSAPlayer::resample(const float* data, size_t len) {
    size_t frames_in = len / channels;
    size_t frames_out = static_cast<size_t>(frames_in / speed);

    buffer.resize(frames_out * channels);

    for (size_t i = 0; i < frames_out; ++i) {
        float pos = i * speed;
        size_t idx = static_cast<size_t>(pos);
        float frac = pos - idx;

        if (idx + 1 >= frames_in)
            idx = frames_in - 2;

        for (unsigned int ch = 0; ch < channels; ++ch) {
            float s1 = data[idx * channels + ch];
            float s2 = data[(idx + 1) * channels + ch];
            float sample = s1 + (s2 - s1) * frac;

            buffer[i * channels + ch] = sample * volume;
        }
    }
}
