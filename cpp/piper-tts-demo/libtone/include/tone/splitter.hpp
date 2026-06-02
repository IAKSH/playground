#pragma once

#include <string>
#include <vector>

// piper在长句上的表现糟糕，建议切句

class SentenceSplitter {
public:
    std::vector<std::string> split(const std::string& text);
};
