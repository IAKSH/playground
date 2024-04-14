#include <fstream>
#include <vector>
#include <iostream>
#include <format>
#include <lz4.h>

// 将string改为C-Style char* 或者C++ iterator以提高泛用性

// 压缩string到内存
std::vector<char> compressData(const std::string& data) {
    int max_compressed_size = LZ4_compressBound(data.size());
    std::vector<char> compressed_data(max_compressed_size + sizeof(int));
    
    // 为了方便解压时确定原始数据长度，这里偏移了一个sizeof(int)再写入的压缩内容，并且在这个int里存了原始数据长度
    int compressed_size = LZ4_compress_default(data.c_str(), compressed_data.data() + sizeof(int), data.size(), max_compressed_size);
    *(int*)compressed_data.data() = data.size();

    compressed_data.resize(compressed_size + sizeof(int));
    return compressed_data;
}

// 解压缩到string
std::string decompressData(const std::vector<char>& compressed_data) {
    // 取出之前塞在开头的表示原始数据长度的int
    int ori_size = *(int*)compressed_data.data();
    std::string decompressed_data(ori_size, ' ');
    LZ4_decompress_safe(compressed_data.data() + sizeof(int), &decompressed_data[0], compressed_data.size() - sizeof(int), ori_size);
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

    std::cout << "input your string and it will be compress to ./output.lz4\n";
    std::string input;
    std::cin >> input;
    compressDataToFile("output.lz4",input);

    std::cout << "input your lz4 file path to read\n";
    std::string reading_path;
    std::cin >> reading_path;
    std::cout << decompressDataFromFile(reading_path);

    return 0;
}