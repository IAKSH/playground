// https://www.lanqiao.cn/problems/2409/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023

#include <bits/stdc++.h>

using namespace std;

array<int,30> nums{
    99,22,51,63,72,61,20,88,40,21,63,30,11,18,99,12,93,16,7,53,64,9,28,84,34,96,52,82,51,77
};

int main() noexcept {
    int cnt = 0;
    for(int i = 0;i < 30;i++) {
        for(int j = i + 1;j < 30;j++) {
            cnt += (nums[i] * nums[j] >= 2022);
        }
    }
    cout << cnt << '\n';
    return 0;
}