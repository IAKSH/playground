#include <iostream>
#include <bitset>
#include <sstream>

// 基本上就是x进制（字符串）到十进制用dec = std::stoi(val,nullptr,x);
// 从十进制转到x进制用ostringstream加上std::ios的进制状态

int main() {
    // 二进制到十进制
    std::string binary = "1010";
    int decimal = std::stoi(binary, nullptr, 2);
    std::cout << decimal << std::endl;  // 输出: 10

    // 十进制到二进制
    std::bitset<32> binaryBits(decimal);
    std::cout << binaryBits << std::endl;  // 输出: 00000000000000000000000000001010

    // 八进制到十进制
    std::string oct = "162";
    std::cout << std::stoi(oct,nullptr,8) << std::endl; // 114

    // 十进制到八进制
    std::ostringstream oss_oct;
    oss_oct << std::oct << 114;
    std::cout << oss_oct.str() << std::endl;  // 输出: 162

    // 十六进制到十进制
    std::string hex = "F3FA";
    std::cout << std::stoi(hex,nullptr,16) << std::endl; // 62458

    // 十进制到十六进制
    std::ostringstream oss;
    oss << std::hex << 62458;
    std::cout << oss.str() << std::endl;  // 输出: f3fa

    return 0;
}
