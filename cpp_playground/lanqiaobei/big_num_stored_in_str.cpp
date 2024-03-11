#include <iostream>
#include <string>

int big_mod(std::string num, int a) {
    int res = 0;
    for (int i = 0; i < num.length(); i++) {
        res = (res*10 + (int)num[i] - '0') % a;
    }
    return res;
}

int main() {
    std::string num = "12345678901234567890123456789012345678901234567890";
    int a = 2023;
    int result = big_mod(num, a);
    std::cout << num << " % " << a << " = " << result << '\n';
    return 0;
}
