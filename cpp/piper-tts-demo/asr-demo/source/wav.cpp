#include <asr/wav.hpp>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <spdlog/spdlog.h>

WavAudio::WavAudio(std::string_view path) {
    parse(path);
}

float WavAudio::duration_ms() const {
    if (m_sample_rate == 0) return 0.0f;
    return (static_cast<float>(m_samples.size()) / m_channels / m_sample_rate) * 1000.0f;
}

void WavAudio::parse(std::string_view path) {
    std::ifstream ifs(path.data(), std::ios::binary);
    if (!ifs) {
        throw std::runtime_error(std::string("Failed to open WAV file: ") + std::string(path));
    }

    // --- Read RIFF header ---
    RiffHeader riff{};
    ifs.read(reinterpret_cast<char*>(&riff), sizeof(riff));
    if (!ifs || std::strncmp(riff.chunk_id, "RIFF", 4) != 0) {
        throw std::runtime_error("Not a valid RIFF file");
    }
    if (std::strncmp(riff.format, "WAVE", 4) != 0) {
        throw std::runtime_error("Not a valid WAVE file");
    }

    bool has_fmt  = false;
    bool has_data = false;

    // --- Scan sub-chunks ---
    while (!has_data && ifs) {
        ChunkHeader chunk{};
        ifs.read(reinterpret_cast<char*>(&chunk), sizeof(chunk));
        if (!ifs) break;

        spdlog::debug("WAV chunk: {:.4} ({} bytes)", chunk.id, chunk.size);

        if (std::strncmp(chunk.id, "fmt ", 4) == 0) {
            FmtChunkData fmt{};
            ifs.read(reinterpret_cast<char*>(&fmt), sizeof(fmt));
            if (!ifs) throw std::runtime_error("Failed to read fmt chunk");

            // Skip any extra fmt bytes beyond the standard 16
            if (chunk.size > sizeof(fmt)) {
                ifs.ignore(chunk.size - sizeof(fmt));
            }

            if (fmt.audio_format != 1) {
                throw std::runtime_error("Only PCM WAV files are supported (audio_format != 1)");
            }

            m_sample_rate    = fmt.sample_rate;
            m_channels       = fmt.num_channels;
            m_bits_per_sample = fmt.bits_per_sample;
            has_fmt = true;

            spdlog::debug("fmt: {} Hz, {} ch, {} bits, audio_format={}",
                          m_sample_rate, m_channels, m_bits_per_sample, fmt.audio_format);
        }
        else if (std::strncmp(chunk.id, "data", 4) == 0) {
            if (!has_fmt) {
                throw std::runtime_error("Found data chunk before fmt chunk");
            }

            const uint32_t bytes_per_sample = m_bits_per_sample / 8;
            const uint32_t total_samples = chunk.size / bytes_per_sample;

            // Read raw PCM as int16 (only 16-bit supported for now)
            if (m_bits_per_sample != 16) {
                throw std::runtime_error("Only 16-bit PCM is supported");
            }

            std::vector<int16_t> raw(total_samples);
            ifs.read(reinterpret_cast<char*>(raw.data()), chunk.size);
            if (!ifs) throw std::runtime_error("Failed to read data chunk");

            // Convert int16 -> float32
            m_samples.resize(total_samples);
            for (size_t i = 0; i < total_samples; ++i) {
                m_samples[i] = static_cast<float>(raw[i]) / 32768.0f;
            }

            has_data = true;
            spdlog::debug("data: {} samples, {:.1f} ms",
                          total_samples, duration_ms());
        }
        else {
            // Unknown chunk — skip it
            spdlog::debug("Skipping unknown chunk: {:.4}", chunk.id);
            ifs.ignore(chunk.size);
        }
    }

    if (!has_fmt)  throw std::runtime_error("No fmt chunk found in WAV file");
    if (!has_data) throw std::runtime_error("No data chunk found in WAV file");

    // Validate sample rate for whisper compatibility
    if (m_sample_rate != 16000) {
        spdlog::warn("Sample rate is {} Hz (expected 16000 Hz for Whisper). "
                     "Recognition accuracy may be degraded.", m_sample_rate);
    }
}

