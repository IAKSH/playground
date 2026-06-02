#include <piper-tts-demo/wav_writer.hpp>
#include <algorithm>
#include <stdexcept>

// ---- RIFF/WAV 结构体（与 asr-demo/include/asr/wav.hpp 一致） ----

#pragma pack(push, 1)
struct RiffHeader {
    char     chunk_id[4];   // "RIFF"
    uint32_t chunk_size;    // 文件剩余大小 (文件总大小 - 8)
    char     format[4];     // "WAVE"
};

struct ChunkHeader {
    char     id[4];
    uint32_t size;          // chunk 数据大小 (不含 header 8 字节)
};

struct FmtChunkData {
    uint16_t audio_format;      // 1 = PCM
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;         // sample_rate * channels * bits_per_sample / 8
    uint16_t block_align;       // channels * bits_per_sample / 8
    uint16_t bits_per_sample;
};
#pragma pack(pop)

// ----------------------------------------------------------------

WavWriter::WavWriter(std::string_view filepath,
                     int sample_rate,
                     uint16_t channels,
                     uint16_t bits_per_sample)
    : m_filepath(filepath)
    , m_sample_rate(sample_rate)
    , m_channels(channels)
    , m_bits_per_sample(bits_per_sample)
{
    m_file.exceptions(std::ios::failbit | std::ios::badbit);
}

WavWriter::~WavWriter() {
    if (!m_finalized) {
        try {
            finalize();
        } catch (...) {
            // 析构函数不抛异常
        }
    }
}

void WavWriter::write(const float* samples, size_t count) {
    if (m_finalized) return;

    for (size_t i = 0; i < count; ++i) {
        // float32 [-1.0, 1.0] -> int16 [-32768, 32767]
        float clamped = std::clamp(samples[i], -1.0f, 1.0f);
        m_buffer.push_back(static_cast<int16_t>(clamped * 32767.0f));
    }
}

void WavWriter::finalize() {
    if (m_finalized) return;
    m_finalized = true;

    m_file.open(m_filepath, std::ios::binary);

    const uint32_t bytes_per_sample = m_bits_per_sample / 8;
    const uint32_t data_size = static_cast<uint32_t>(m_buffer.size() * bytes_per_sample);

    // 1) RIFF header
    RiffHeader riff{};
    riff.chunk_id[0] = 'R'; riff.chunk_id[1] = 'I';
    riff.chunk_id[2] = 'F'; riff.chunk_id[3] = 'F';
    riff.chunk_size = static_cast<uint32_t>(36 + data_size); // 36 = sizeof(fmt_chunk) + sizeof(fmt_data)
    riff.format[0] = 'W'; riff.format[1] = 'A';
    riff.format[2] = 'V'; riff.format[3] = 'E';
    m_file.write(reinterpret_cast<const char*>(&riff), sizeof(riff));

    // 2) fmt  chunk ("fmt ")
    ChunkHeader fmt_header{};
    fmt_header.id[0] = 'f'; fmt_header.id[1] = 'm';
    fmt_header.id[2] = 't'; fmt_header.id[3] = ' ';
    fmt_header.size = sizeof(FmtChunkData);
    m_file.write(reinterpret_cast<const char*>(&fmt_header), sizeof(fmt_header));

    FmtChunkData fmt{};
    fmt.audio_format    = 1;  // PCM
    fmt.num_channels    = m_channels;
    fmt.sample_rate     = static_cast<uint32_t>(m_sample_rate);
    fmt.byte_rate       = static_cast<uint32_t>(m_sample_rate * m_channels * bytes_per_sample);
    fmt.block_align     = m_channels * bytes_per_sample;
    fmt.bits_per_sample = m_bits_per_sample;
    m_file.write(reinterpret_cast<const char*>(&fmt), sizeof(fmt));

    // 3) data chunk
    ChunkHeader data_header{};
    data_header.id[0] = 'd'; data_header.id[1] = 'a';
    data_header.id[2] = 't'; data_header.id[3] = 'a';
    data_header.size = data_size;
    m_file.write(reinterpret_cast<const char*>(&data_header), sizeof(data_header));

    m_file.write(reinterpret_cast<const char*>(m_buffer.data()),
                 static_cast<std::streamsize>(data_size));

    m_file.close();
}
