#include <fstream>
#include <vector>
#include <iostream>
#include <format>
#include <lzma.h>

// 压缩string到内存
std::vector<char> compressData(const std::string& data) {
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_easy_encoder(&strm, LZMA_PRESET_DEFAULT, LZMA_CHECK_CRC64);

    std::vector<char> compressed_data(data.size() + LZMA_BLOCK_HEADER_SIZE_MAX);
    strm.next_in = (const uint8_t*)data.data();
    strm.avail_in = data.size();
    strm.next_out = (uint8_t*)compressed_data.data();
    strm.avail_out = compressed_data.size();

    while (strm.avail_in != 0) {
        ret = lzma_code(&strm, LZMA_RUN);
    }

    ret = lzma_code(&strm, LZMA_FINISH);

    compressed_data.resize(strm.total_out);
    lzma_end(&strm);
    return compressed_data;
}

// 解压缩到string
std::string decompressData(const std::vector<char>& compressed_data) {
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);

    std::string decompressed_data(compressed_data.size(), ' ');
    strm.next_in = (const uint8_t*)compressed_data.data();
    strm.avail_in = compressed_data.size();
    strm.next_out = (uint8_t*)decompressed_data.data();
    strm.avail_out = decompressed_data.size();

    while (strm.avail_in != 0) {
        ret = lzma_code(&strm, LZMA_RUN);
    }

    decompressed_data.resize(decompressed_data.size() - strm.avail_out);
    lzma_end(&strm);
    return decompressed_data;
}

// 压缩string到文件
void compressDataToFile(const std::string& filename, const std::string& data) {
    std::vector<char> compressed_data = compressData(data);
    std::ofstream file(filename, std::ios::binary);
    file.write(compressed_data.data(), compressed_data.size());
}

// 从文件中解压缩到string
std::string decompressDataFromFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    std::vector<char> compressed_data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    return decompressData(compressed_data);
}

int main() noexcept {
    std::string ori = R"(
        Beautiful is better than ugly.
        Explicit is better than implicit.
        Simple is better than complex.
        Complex is better than complicated.
        Flat is better than nested.
        Sparse is better than dense.
        Readability counts.
        Special cases aren't special enough to break the rules.
        Although practicality beats purity.
        Errors should never pass silently.
        Unless explicitly silenced.
        In the face of ambiguity, refuse the temptation to guess.
        There should be one-- and preferably only one --obvious way to do it.[c]
        Although that way may not be obvious at first unless you're Dutch.
        Now is better than never.
        Although never is often better than right now.[d]
        If the implementation is hard to explain, it's a bad idea.
        If the implementation is easy to explain, it may be a good idea.
        Namespaces are one honking great idea - let's do more of those!
    )";

    auto compressed = compressData(ori);

    std::cout << std::format("before:\t{} Byte\n",ori.size() * sizeof(char));
    std::cout << std::format("after:\t{} Byte\n",compressed.size() * sizeof(char));

    std::cout << "input your string and it will be compress to ./output.lzma\n";
    std::string input;
    std::cin >> input;
    compressDataToFile("output.lzma",input);

    std::cout << "input your lzma file path to read\n";
    std::string reading_path;
    std::cin >> reading_path;
    std::cout << decompressDataFromFile(reading_path);

    return 0;
}
