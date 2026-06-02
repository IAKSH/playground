#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <spdlog/spdlog.h>
#include <whisper.h>
#include <asr/wav.hpp>
#include "common-sdl.h"

// ============================================================================
//  Constants
// ============================================================================

static const char* DEFAULT_MODEL   = "../../model/whisper/ggml-tiny-q8_0.bin";
static const char* DEFAULT_WAV     = "../../asr-demo/sounds/output.wav";
//static const int   STEP_MS         = 3000;   // process every 3 seconds
static const int   STEP_MS         = 1000;
static const int   LENGTH_MS       = 10000;  // keep 10 seconds of audio context
static const int   KEEP_MS         = 200;    // overlap between steps

// Voice Activity Detection parameters
static const float VAD_THOLD       = 0.6f;   // energy ratio threshold (lower = more sensitive)
static const float FREQ_THOLD      = 100.0f; // high-pass filter cutoff (Hz)

// ============================================================================
//  Simple Voice Activity Detection (adapted from whisper.cpp common.cpp)
//  Returns true if the last `last_ms` of audio is likely silence (no speech).
// ============================================================================

static void high_pass_filter(std::vector<float>& data, float cutoff, float sample_rate) {
    const float rc    = 1.0f / (2.0f * M_PI * cutoff);
    const float dt    = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];
    for (size_t i = 1; i < data.size(); ++i) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

static bool is_silence(std::vector<float>& pcmf32, int sample_rate,
                       int last_ms, float vad_thold, float freq_thold) {
    const int n_samples      = static_cast<int>(pcmf32.size());
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        return true;  // not enough samples → treat as silence
    }

    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }

    float energy_all  = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; ++i) {
        energy_all += std::fabs(pcmf32[i]);
        if (i >= n_samples - n_samples_last) {
            energy_last += std::fabs(pcmf32[i]);
        }
    }

    energy_all  /= n_samples;
    energy_last /= n_samples_last;

    // recent energy ≤ threshold * overall → silence
    return (energy_last <= vad_thold * energy_all);
}

// ============================================================================
//  File-based ASR (--wav mode)
// ============================================================================

static int run_file_asr(const char* wav_path, const char* model_path) {
    spdlog::info("Mode: file-based ASR");
    spdlog::info("WAV file: {}", wav_path);
    spdlog::info("Model:    {}", model_path);

    // 1. Load WAV
    WavAudio audio(wav_path);
    spdlog::info("Loaded: {} Hz, {} ch, {} bits, {:.1f} ms, {} samples",
                 audio.sample_rate(), audio.channels(),
                 audio.bits_per_sample(), audio.duration_ms(),
                 audio.samples().size());

    // 2. Load whisper model
    whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        spdlog::error("Failed to load model: {}", model_path);
        return 1;
    }

    // 3. Configure & run inference
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.language         = "zh";
    params.print_progress   = true;
    params.print_realtime   = false;
    params.print_timestamps = true;
    params.no_timestamps    = false;

    int ret = whisper_full(ctx, params,
                           audio.samples().data(),
                           static_cast<int>(audio.samples().size()));
    if (ret != 0) {
        spdlog::error("whisper_full failed: {}", ret);
        whisper_free(ctx);
        return ret;
    }

    // 4. Print results
    const int n = whisper_full_n_segments(ctx);
    spdlog::info("{} segment(s)", n);
    for (int i = 0; i < n; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);
        int64_t t0 = whisper_full_get_segment_t0(ctx, i);
        int64_t t1 = whisper_full_get_segment_t1(ctx, i);
        spdlog::info("[{:02d}:{:02d}.{:03d} -> {:02d}:{:02d}.{:03d}] {}",
                     static_cast<int>(t0 / 6000),
                     static_cast<int>((t0 / 100) % 60),
                     static_cast<int>(t0 % 100) * 10,
                     static_cast<int>(t1 / 6000),
                     static_cast<int>((t1 / 100) % 60),
                     static_cast<int>(t1 % 100) * 10,
                     text);
    }

    whisper_free(ctx);
    return 0;
}

// ============================================================================
//  Real-time microphone ASR (default mode)
// ============================================================================

static int run_mic_asr(const char* model_path) {
    spdlog::info("Mode: real-time microphone ASR");
    spdlog::info("Model:  {}", model_path);
    spdlog::info("Window: step={}ms  length={}ms  keep={}ms", STEP_MS, LENGTH_MS, KEEP_MS);

    const int n_samples_step = (STEP_MS   * WHISPER_SAMPLE_RATE) / 1000;
    const int n_samples_len  = (LENGTH_MS * WHISPER_SAMPLE_RATE) / 1000;
    const int n_samples_keep = (KEEP_MS   * WHISPER_SAMPLE_RATE) / 1000;
    const int n_samples_30s  = (30000     * WHISPER_SAMPLE_RATE) / 1000;

    // 1. Init microphone capture (SDL2)
    audio_async mic(LENGTH_MS);
    if (!mic.init(/*capture_id=*/-1, WHISPER_SAMPLE_RATE)) {
        spdlog::error("Failed to initialize microphone");
        return 1;
    }
    mic.resume();
    spdlog::info("Microphone ready — start speaking!");

    // 2. Load whisper model
    whisper_context_params cparams = whisper_context_default_params();
    struct whisper_context* ctx = whisper_init_from_file_with_params(model_path, cparams);
    if (!ctx) {
        spdlog::error("Failed to load model: {}", model_path);
        return 1;
    }

    // Audio buffers
    std::vector<float> pcmf32(n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    bool is_running = true;
    spdlog::info("Press Ctrl+C to stop.\n");

    // 3. Main streaming loop
    while (is_running) {
        // Handle SDL events (Ctrl+C / window close)
        is_running = sdl_poll_events();
        if (!is_running) break;

        // Wait for enough audio
        while (true) {
            is_running = sdl_poll_events();
            if (!is_running) break;

            mic.get(STEP_MS, pcmf32_new);

            if (static_cast<int>(pcmf32_new.size()) > 2 * n_samples_step) {
                spdlog::warn("Processing too slow, dropping audio...");
                mic.clear();
                continue;
            }

            if (static_cast<int>(pcmf32_new.size()) >= n_samples_step) {
                mic.clear();
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!is_running) break;

        const int n_new = static_cast<int>(pcmf32_new.size());

        // Combine: keep old context + new audio
        const int n_take = std::min(
            static_cast<int>(pcmf32_old.size()),
            std::max(0, n_samples_keep + n_samples_len - n_new));

        pcmf32.resize(n_new + n_take);

        for (int i = 0; i < n_take; ++i) {
            pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_take + i];
        }
        std::memcpy(pcmf32.data() + n_take, pcmf32_new.data(),
                    n_new * sizeof(float));

        pcmf32_old = pcmf32;

        // 4. Skip inference if silence (energy-based VAD)
        if (is_silence(pcmf32, WHISPER_SAMPLE_RATE, 1000, VAD_THOLD, FREQ_THOLD)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // 5. Run inference
        {
            whisper_full_params wparams =
                whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_realtime   = false;
            wparams.print_timestamps = true;
            wparams.single_segment   = false;
            wparams.max_tokens       = 0;
            wparams.language         = "zh";
            wparams.no_context       = true;    // key fix: prevent hallucination propagation
            wparams.no_speech_thold  = 0.01f;   // lower threshold to avoid suppressing real speech

            if (whisper_full(ctx, wparams, pcmf32.data(),
                             static_cast<int>(pcmf32.size())) != 0) {
                spdlog::error("whisper_full failed");
                break;
            }

            // Print only the latest segment (incremental output)
            {
                const int n_segments = whisper_full_n_segments(ctx);
                if (n_segments > 0) {
                    const int i = n_segments - 1;
                    const char* text = whisper_full_get_segment_text(ctx, i);
                    if (text && text[0] != '\0') {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);
                        fmt::print("[{:02d}:{:02d}.{:03d} -> {:02d}:{:02d}.{:03d}] {}\n",
                                   static_cast<int>(t0 / 6000),
                                   static_cast<int>((t0 / 100) % 60),
                                   static_cast<int>(t0 % 100) * 10,
                                   static_cast<int>(t1 / 6000),
                                   static_cast<int>((t1 / 100) % 60),
                                   static_cast<int>(t1 % 100) * 10,
                                   text);
                    }
                }
            }
        }
    }

    // Cleanup
    mic.pause();
    whisper_free(ctx);
    spdlog::info("Microphone ASR stopped.");
    return 0;
}

// ============================================================================
//  Main — parse --wav argument and dispatch
// ============================================================================

static void print_usage(const char* prog) {
    fmt::print("Usage:\n");
    fmt::print("  {}                  Real-time microphone ASR (default)\n", prog);
    fmt::print("  {} --wav <file>     ASR from a WAV file\n", prog);
    fmt::print("\nOptions:\n");
    fmt::print("  --model <path>      Whisper model path (default: {})\n", DEFAULT_MODEL);
}

int main(int argc, char** argv) {
    spdlog::set_level(spdlog::level::debug);

    std::string wav_path;
    std::string model_path = DEFAULT_MODEL;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--wav") == 0 && i + 1 < argc) {
            wav_path = argv[++i];
        } else if (std::strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fmt::print(stderr, "Unknown argument: {}\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // Dispatch
    if (!wav_path.empty()) {
        return run_file_asr(wav_path.c_str(), model_path.c_str());
    } else {
        return run_mic_asr(model_path.c_str());
    }
}