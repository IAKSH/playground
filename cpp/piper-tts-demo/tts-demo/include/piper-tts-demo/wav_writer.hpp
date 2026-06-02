#pragma once

#include <cstdint>
#include <string_view>
#include <vector>
#include <fstream>

/// 将 float32 PCM 样本写入标准 16-bit PCM WAV 文件
class WavWriter {
public:
    /// @param filepath  输出文件路径
    /// @param sample_rate  采样率 (Hz)，从 Piper 的 audio chunk 中获取
    /// @param channels  声道数，默认 1 (单声道)
    /// @param bits_per_sample  位深，默认 16
    WavWriter(std::string_view filepath,
              int sample_rate,
              uint16_t channels = 1,
              uint16_t bits_per_sample = 16);

    ~WavWriter();

    WavWriter(const WavWriter&) = delete;
    WavWriter& operator=(const WavWriter&) = delete;

    /// 追加 float32 PCM 样本，内部转换为 int16 缓存
    void write(const float* samples, size_t count);

    /// 写入 WAV 文件头和数据，关闭文件
    void finalize();

    /// 是否已经关闭
    bool finalized() const { return m_finalized; }

    /// 已写入的样本数
    size_t total_samples() const { return m_buffer.size(); }

private:
    void write_header();

    std::string    m_filepath;
    std::ofstream  m_file;
    std::vector<int16_t> m_buffer;

    int      m_sample_rate;
    uint16_t m_channels;
    uint16_t m_bits_per_sample;
    bool     m_finalized = false;
};
