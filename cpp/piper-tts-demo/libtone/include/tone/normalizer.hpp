#pragma once
#include <string_view>
#include <vector>
#include <regex>

class TextNormalizer
{
public:
    TextNormalizer();

    std::string normalize(std::string_view text);

private:

    std::string remove_markdown(std::string_view text);
    std::string normalize_url(std::string_view text);
    std::string normalize_email(std::string_view text);
    std::string normalize_symbols(std::string_view text);
    std::string normalize_acronyms(std::string_view text);
    //std::string normalize_numbers(std::string_view text);
    std::string collapse_spaces(std::string_view text);

    std::string tokenize_english(std::string_view text);
    std::string tokenize_numbers(std::string_view text);

    std::regex acronym_pattern;
    std::regex number_pattern;
};