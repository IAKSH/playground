/**
 * https://www.lanqiao.cn/problems/2097/learning/?page=2&first_category_id=1&second_category_id=3&tags=2022
 * 也许是前缀合（核心算法） + 二分查找（时间复杂度优化）
 * 任何一个长度为y的区间都要至少能被踩2x次
 * 求y_min
*/

#ifdef OLD_VER
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
#else
#include <bits/stdc++.h>

using namespace std;

static int n,x;
static vector<int> v,prefix;

bool check(int y) noexcept {
    int sum = 0;
    for(int i = 0;i < n - y;i++) {
        //sum = accumulate(v.begin() + i,v.begin() + i + y,0);
        // 使用预先计算的前缀和，减少重复的遍历求和
        sum = prefix[i + y - 1] - (i > 0 ? prefix[i - 1] : 0); 
        if(sum < 2 * x) {
            return false;
        }
    }
    return true;
}

int main() noexcept {
    cin >> n >> x;
    v.resize(n - 1);
    prefix.resize(n - 1);
    for(int i = 0; i < n - 1; i++) {
        cin >> v[i];
        // 输入的同时计算前缀和
        prefix[i] = v[i] + (i > 0 ? prefix[i - 1] : 0);
    }

    // 在1到n-1之间做二分查找
    // n可能会非常大，不太适合使用std::lower_bound，因为需要为其构建一个长度为n的range
    int l = 0;
    int r = n;
    int mid;
    while(l != r) {
        mid = (r + l) / 2;
        if(check(mid))
            r = mid;
        else
            l = mid + 1;
    }
    cout << r << '\n';
    
    return 0;
}
#endif