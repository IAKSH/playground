// https://www.lanqiao.cn/problems/11014/learning/?page=1

/**
 * eg:
 * 2345 = 
 * 1111 +
 * 1111 +
 * 0111 +
 * 0011 +
 * 0001
 * 
 * cnt = 5
 * equal to the largest digit of this number
*/

#include <bits/stdc++.h>

using namespace std;

int foo(int n) noexcept {
    int res = 0;
    while(n > 0) {
        n /= 10;
        ++res;
    }
    return res;
}

int main() noexcept {
    int n;
    cin >> n;

    int maxn = 0;
    for(int i = 0;i < foo(n);i++) {
        maxn = max(maxn,n / static_cast<int>(pow(10,i)) % 10);
    }
    cout << maxn << '\n';
    return 0;
}