// https://www.lanqiao.cn/problems/3498/learning/?page=1&first_category_id=1&difficulty=20&tags=2023
// 未经测试

#include <bits/stdc++.h>

using namespace std;

#ifdef USE_BITSET
template <size_t LEN>
pair<int,int> countBits(bitset<LEN> bits) noexcept {
    int t = 0;
    int f = 0;
    for(int i = 0;i < LEN;i++)
        ++(bits[i] ? t : f);
    return pair<int,int>(t,f);
}

template <size_t LEN>
int shannonEntropy(bitset<LEN> bits) noexcept {
    auto cnt = countBits(bits);
    float t_ratio = static_cast<float>(cnt.first) / (cnt.first + cnt.second);
    float f_ratio = 1.0f - t_ratio;
    int res = 0;
    for(int i = 1;i < LEN;i++) {
        if(bits[i])
            res += t_ratio * log2f(t_ratio);
        else
            res += f_ratio * log2f(f_ratio);
    }
    return -res;
}

template <size_t LEN>
void binaryIncrease(bitset<LEN> bits) noexcept {
    bool b = bits[LEN - 1];
    bits[LEN - 1] = !bits[LEN - 1];
    if(b) {
        bool carry = true;
        for(int i = LEN - 2;carry && i >= 0;i--) {
            carry = bits[i];
            bits[i] = !bits[i];
        }
    }
}

int main() noexcept {
    bitset<23333333> bits{false};
    while(!bits.all()) {
        int t_count = bits.count();
        int f_count = 23333333 - bits.count();
        if(t_count > f_count && shannonEntropy(bits) == 11625907.5798f) {
            cout << f_count << '\n';
            return 0;
        }
        binaryIncrease(bits);
    }
    return 1;
}
#else
pair<int,int> countBits(const vector<bool>& bits) noexcept {
    int t = 0;
    int f = 0;
    for(const auto& b : bits)
        ++(b ? t : f);
    return pair<int,int>(t,f);
}

int shannonEntropy(const vector<bool>& bits) noexcept {
    auto cnt = countBits(bits);
    float t_ratio = static_cast<float>(cnt.first) / (cnt.first + cnt.second);
    float f_ratio = 1.0f - t_ratio;
    int res = 0;
    int len = bits.size();
    for(int i = 1;i < len;i++) {
        if(bits[i])
            res += t_ratio * log2f(t_ratio);
        else
            res += f_ratio * log2f(f_ratio);
    }
    return -res;
}

void binaryIncrease(vector<bool>& bits) noexcept {
    int len = bits.size();
    bool b = bits[len - 1];
    bits[len - 1] = !bits[len - 1];
    if(b) {
        bool carry = true;
        for(int i = len - 2;carry && i >= 0;i--) {
            carry = bits[i];
            bits[i] = !bits[i];
        }
    }
}

bool checkAllT(const vector<bool>& bits) noexcept {
    for(const auto& b : bits)
        if(!b) return false;
    return true;
}

int main() noexcept {
    vector<bool> bits(23333333,false);
    while(!checkAllT(bits)) {
        auto count = countBits(bits);
        if(count.first > count.second && shannonEntropy(bits) == 11625907.5798f) {
            cout << count.second << '\n';
            return 0;
        }
        binaryIncrease(bits);
    }
    return 1;
}
#endif