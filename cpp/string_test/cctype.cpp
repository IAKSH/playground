#include <iostream>
#include <algorithm>
#include <string>
#include <cctype>

// cctype是C头文件ctype.h (char type)的C++版
// 提供了针对单个字符的一些识别和转换功能

/*
namespace std
{
  using ::isalnum;  是否是字母(alpha)或数字
  using ::isalpha;
  using ::iscntrl;  是否是控制字符
  using ::isdigit;  是否是十进制数
  using ::isgraph;  是否有图形表示法
  using ::islower;  是否是小写字母
  using ::isprint;  是否可打印
  using ::ispunct;  是否是标点符号
  using ::isspace;  是否是空格
  using ::isupper;  是否是大写字母
  using ::isxdigit; 是否是十进制数字
  using ::tolower;  大写字母转小写
  using ::toupper;  小写字母转大写
} // namespace std
*/

/*
#if __cplusplus >= 201103L
#ifdef _GLIBCXX_USE_C99_CTYPE_TR1
#undef isblank
namespace std
{
  using ::isblank;  检查字符是否被本地环境分类为空格字符
} // namespace std
*/

int main() noexcept {
    std::string str{"1nihao"};
    std::cout << (std::isdigit(str[0]) ? "true" : "false") << std::endl;
}
