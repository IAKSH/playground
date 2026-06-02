#include <tone/normalizer.hpp>
#include <unordered_map>

TextNormalizer::TextNormalizer()
: acronym_pattern(R"(\b([A-Z]{2,})\b)"),
  number_pattern(R"(\d{2,})")
{}

std::string TextNormalizer::normalize(std::string_view text)
{
    std::string t(text);
    t = remove_markdown(t);
    t = normalize_url(t);
    t = normalize_email(t);
    t = normalize_symbols(t);
    t = normalize_acronyms(t);

    t = tokenize_english(t);
    t = tokenize_numbers(t);

    //t = normalize_numbers(t);
    t = collapse_spaces(t);
    return t;
}

std::string TextNormalizer::remove_markdown(std::string_view text) {
    std::string t(text);
    t = std::regex_replace(t,std::regex("`([^`]*)`"),"$1");
    t = std::regex_replace(t,std::regex("\\*\\*([^*]+)\\*\\*"),"$1");
    return t;
}

std::string TextNormalizer::normalize_url(std::string_view text) {
    std::string t(text);
    t = std::regex_replace(t,std::regex(R"(https?://)"),"https 斜杠斜杠 ");
    t = std::regex_replace(t,std::regex(R"(www\.)"),"www 点 ");
    t = std::regex_replace(t,std::regex(R"(\.com)")," 点 com");
    t = std::regex_replace(t,std::regex(R"(\.cn)")," 点 cn");
    return t;
}

std::string TextNormalizer::normalize_email(std::string_view text) {
    return std::regex_replace(std::string(text),std::regex("@")," 艾特 ");
}

std::string TextNormalizer::normalize_symbols(std::string_view text) {
    static const std::unordered_map<std::string,std::string> symbols {
        {"#"," 井号 "},
        {"%"," 百分号 "},
        {"&"," 和 "},
        {"_"," 下划线 "},
        {"/"," 斜杠 "}
    };

    std::string t(text);
    for(auto& [k,v] : symbols)
        t = std::regex_replace(t,std::regex("\\"+k),v);

    return t;
}

std::string TextNormalizer::normalize_acronyms(std::string_view text) {
    std::string result;
    std::string s(text);

    auto begin = std::sregex_iterator(s.begin(),s.end(),acronym_pattern);
    auto end = std::sregex_iterator();

    size_t last = 0;

    for(auto i=begin;i!=end;++i) {
        const auto& m=*i;
        result.append(text,last,m.position()-last);
        std::string s=m.str();

        for(char c:s) {
            result+=c;
            result+=" ";
        }

        last=m.position()+m.length();
    }

    result.append(text,last,text.size()-last);
    return result;
}

/*
std::string TextNormalizer::normalize_numbers(std::string_view text) {
    std::string result;
    std::string s(text);

    auto begin = std::sregex_iterator(s.begin(),s.end(),number_pattern);
    auto end = std::sregex_iterator();

    size_t last=0;

    for(auto i=begin;i!=end;++i) {
        const auto& m=*i;
        result.append(text,last,m.position()-last);
        std::string s=m.str();
        for(char c:s) {
            result+=c;
            result+=" ";
        }

        last=m.position()+m.length();
    }

    result.append(text,last,text.size()-last);
    return result;
}
*/

std::string TextNormalizer::tokenize_numbers(std::string_view text) {
    std::string result;
    std::string s(text);
    std::regex pattern(R"(\d+(\.\d+)?)");

    auto begin = std::sregex_iterator(s.begin(),s.end(),pattern);
    auto end = std::sregex_iterator();

    size_t last=0;

    for(auto i=begin;i!=end;++i) {
        const auto& m=*i;
        result.append(text,last,m.position()-last);
        std::string s=m.str();
        for(char c:s) {
            if(c=='.')
                result+=" 点 ";
            else {
                result+=c;
                result+=" ";
            }
        }

        last=m.position()+m.length();
    }

    result.append(text,last,text.size()-last);
    return result;
}

std::string TextNormalizer::collapse_spaces(std::string_view text) {
    return std::regex_replace(std::string(text),std::regex(R"(\s+)")," ");
}

std::string TextNormalizer::tokenize_english(std::string_view text) {
    std::string t(text);
    // 尝试按大驼峰法进行分词
    t = std::regex_replace(
        t,
        std::regex(R"(([a-z])([A-Z]))"),
        "$1 $2"
    );
    // 英文单词两侧加空格
    t = std::regex_replace(
        t,
        std::regex(R"(([A-Za-z]+))"),
        " $1 "
    );

    return t;
}
