/**
 * https://www.lanqiao.cn/problems/2097/learning/?page=2&first_category_id=1&second_category_id=3&tags=2022
 * 也许是前缀合（核心算法） + 二分查找（时间复杂度优化）
 * 任何一个长度为y的区间都要至少能被踩2x次
 * 求y_min
*/

#include <bits/stdc++.h>

using namespace std;

static int n,x;
static vector<int> v;

/**
 * 先来个线性查找的
 * 3个超时
*/
int main() noexcept {
    cin >> n >> x;
    v.resize(n - 1);
    for(auto& i : v)
        cin >> i;

    int y;
    for(y = 1;true;y++) {
        bool flag = true;
        for(int i = 0;i < n - y;i++) {
            int sum = accumulate(v.begin() + i,v.begin() + i + y,0);
            if(sum < 2 * x) {
                flag = false;
                break;
            }
        }
        if(flag) {
            cout << y << '\n';
            return 0;
        }
    }

    return 1;
}