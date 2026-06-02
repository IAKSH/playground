#pragma once
#include <cstdint>
#include <string_view>
#include <vector>

// RIFF/WAV file format structures for chunk-based parsing
struct RiffHeader {
    char     chunk_id[4];   // "RIFF"
    uint32_t chunk_size;    // file size - 8
    char     format[4];     // "WAVE"
};

struct ChunkHeader {
    char     id[4];
    uint32_t size;
};

struct FmtChunkData {
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
};

class WavAudio {
public:
    explicit WavAudio(std::string_view path);

    uint32_t               sample_rate()    const { return m_sample_rate; }
    uint16_t               channels()       const { return m_channels; }
    uint16_t               bits_per_sample() const { return m_bits_per_sample; }
    const std::vector<float>& samples()     const { return m_samples; }
    float                  duration_ms()    const;

private:
    void parse(std::string_view path);

    std::vector<float> m_samples;
    uint32_t           m_sample_rate    = 0;
    uint16_t           m_channels       = 0;
    uint16_t           m_bits_per_sample = 0;
};
