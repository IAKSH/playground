// https://www.lanqiao.cn/problems/3499/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023

#include <bits/stdc++.h>

using namespace std;

bool check_bin(int dec) noexcept {
    return dec % bitset<32>(dec).count() == 0;
}

bool check_oct(int dec) noexcept {
    std::ostringstream oss;
    oss << std::oct << dec;
    int sum = 0;
    char tmp_str[2]{" "};
    for(const auto& c : oss.str()) {
        tmp_str[0] = c;
        sum += stoi(tmp_str,nullptr,8);
    }
    return dec % sum == 0;
}

bool check_hex(int dec) noexcept {
    std::ostringstream oss;
    oss << std::hex << dec;
    int sum = 0;
    char tmp_str[2]{" "};
    for(const auto& c : oss.str()) {
        tmp_str[0] = c;
        sum += stoi(tmp_str,nullptr,16);
    }
    return dec % sum == 0;
}

bool check_dec(int dec) noexcept {
    int sum = 0;
    int n = dec;
    while(n > 0) {
        sum += n % 10;
        n /= 10;
    }
    return dec % sum == 0;
}

int main() noexcept {
    int cnt = 0;
    int i = 1;
    for(;true;i++) {
        cnt += (check_dec(i) && check_hex(i) && check_oct(i) && check_bin(i));
        if(cnt == 2023)
            break;
    }
    cout << i << '\n';
    return 0;
}