// https://www.lanqiao.cn/problems/3496/learning/?page=1&first_category_id=1&difficulty=20&tags=2023

#include <bits/stdc++.h>

using namespace std;

int getLen(int n) noexcept {
    int len = 0;
    while(n > 0) {
        ++len;
        n /= 10;
    }
    return len;
}

bool check(int n) noexcept {
    int len = getLen(n);
    int i,j = len - 1;
    bitset<4> flags;
    array<int,4> nums{2,0,2,3};
    for(i = 0;i < 4;i++) {
        for(;j >= 0;j--) {
            if(n / static_cast<int>(pow(10,j)) % 10 == nums[i]) {
                flags[i] = true;
                break;
            }
        }
    }
    return flags.all();
}

int main() noexcept {
    int cnt = 0;
    for(int i = 12345678;i <= 98765432;i++)
        cnt += !check(i);
    cout << cnt << '\n';
    return 0;
}