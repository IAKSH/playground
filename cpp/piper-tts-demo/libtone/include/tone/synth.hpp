#pragma once

#include <string_view>
#include <piper.h>
#include <functional>

class Synthesizer {
public:
    Synthesizer(Synthesizer&) = delete;
    Synthesizer(std::string_view model_path,std::string_view espeak_path);
    ~Synthesizer();

    void generate(std::string_view s,std::function<void(const float* pcm,size_t pcm_len)> callback);

private:
    void setup_piper(std::string_view model_path,std::string_view espeak_path);
    void release_piper();

    piper_synthesizer *synth;
    piper_synthesize_options options;
    piper_audio_chunk chunk;
};