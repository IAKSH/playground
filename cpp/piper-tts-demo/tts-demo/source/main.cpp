#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>

#include <tone/synth.hpp>
#include <tone/player.hpp>
#include <piper-tts-demo/wav_writer.hpp>

// ---- 模型与资源路径 ----
static constexpr std::string_view MODEL_AMY    {"../../model/piper/en_US-amy-medium.onnx"};
static constexpr std::string_view MODEL_XIAOYA {"../../model/piper/zh_CN-xiao_ya-medium.onnx"};
static constexpr std::string_view MODEL_HUAYAN {"../../model/piper/zh_CN-huayan-medium.onnx"};
static constexpr std::string_view ESPEAK       {"./vendor/espeak-ng/espeak-ng-data"};

// ---- 工具函数 ----

/// 生成递增的 WAV 文件名，如 "output_000.wav"
static std::string make_wav_filename(int index) {
    std::ostringstream oss;
    oss << "output_" << std::setw(3) << std::setfill('0') << index << ".wav";
    return oss.str();
}

/// 合成语音，同时播放和保存到 WAV
/// @param synth      Piper 合成器
/// @param player     ALSA 播放器
/// @param text       待合成的文本
/// @param wav_path   输出 WAV 路径（空字符串表示不保存）
static void synthesize_and_play(Synthesizer& synth,
                                ALSAPlayer& player,
                                std::string_view text,
                                const std::string& wav_path) {
    // 如果指定了路径，创建 WavWriter（先不传采样率，从第一块 chunk 获取）
    WavWriter* writer = nullptr;
    bool sample_rate_set = false;

    if (!wav_path.empty()) {
        // 采样率暂用 22050（Piper 默认），之后从 chunk 动态修正
        writer = new WavWriter(wav_path, 22050);
    }

    player.prepare();

    synth.generate(text, [&](const float* pcm, size_t pcm_len) {
        // 播放
        player.play(pcm, pcm_len);

        // 保存到 WAV
        if (writer) {
            writer->write(pcm, pcm_len);
        }
    });

    player.drain();

    // 完成 WAV 写入
    if (writer) {
        writer->finalize();
        std::cout << "[WAV] saved: " << wav_path
                  << " (" << writer->total_samples() << " samples)\n";
        delete writer;
    }
}

// ---- 主循环 ----
int main(int, char**) {
    Synthesizer synth(MODEL_HUAYAN, ESPEAK);
    ALSAPlayer player;
    std::string input;
    int file_index = 0;

    while (true) {
        std::cout << "Ready for input! (empty to skip save)\n> ";
        std::getline(std::cin, input);

        if (input.empty()) {
            std::cout << "Empty input, exiting.\n";
            break;
        }

        // 为每次合成生成一个递增的文件名
        std::string wav_path = make_wav_filename(file_index++);

        synthesize_and_play(synth, player, input, wav_path);
    }

    return 0;
}
