#include <tone/splitter.hpp>

std::vector<std::string> SentenceSplitter::split(const std::string& text) {
    std::vector<std::string> out;
    std::string current;

    for(char c : text) {
        current += c;
        // TODO：可能要换成字符串匹配。
        if(c=='。' || c=='！' || c=='？' || c=='!' || c=='?') {
            out.push_back(current);
            current.clear();
        }
    }

    if(!current.empty())
        out.push_back(current);

    return out;
}