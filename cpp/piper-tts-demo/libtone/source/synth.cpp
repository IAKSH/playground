#include <tone/synth.hpp>

Synthesizer::Synthesizer(std::string_view model_path,std::string_view espeak_path) {
    setup_piper(model_path,espeak_path);
}

Synthesizer::~Synthesizer() {
    release_piper();
}

void Synthesizer::generate(std::string_view s,std::function<void(const float* pcm,size_t pcm_len)> callback) {
    // piper_synthesize_start 并不会立即开始合成
    // 合成发生在 piper_synthesize_next
    piper_synthesize_start(synth,s.data(), &options);
    while(piper_synthesize_next(synth, &chunk) != PIPER_DONE) {
        callback(chunk.samples,chunk.num_samples);
    }
}

void Synthesizer::setup_piper(std::string_view model_path,std::string_view espeak_path) {
    synth = piper_create(
        model_path.data(),
        nullptr,
        espeak_path.data()
    );
    options = piper_default_synthesize_options(synth);
}

void Synthesizer::release_piper() {
    piper_free(synth);
}
