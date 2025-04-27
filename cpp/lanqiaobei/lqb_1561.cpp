// https://www.lanqiao.cn/problems/1561/learning/?page=5&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0
// 因为是填空题，所以甚至不用考虑用素数筛之类的
// 然后因为纯素数只需要判断每一位属不属于素数，要比判断整个数是不是快很多，所以优先判断纯素数
// 再在纯素数中判断是不是素数，要比反过来快很多

#include <bits/stdc++.h>

using namespace std;

// 带优化的暴力素数判断
bool is_prime(int n) {
    if(n < 2)
        return 0;
    for(int i = 2;i * i <= n;i++)
        if(n % i == 0)
            return false;
    return true;
}

bool is_pure(int n) {
    int i;
    while(n > 0) {
        i = n % 10;
        if(i == 0 || i == 1 || i == 4 || i == 6 || i == 8 || i == 9)
            return false;
        n /= 10;
    }
    return true;
}

int main() {
    int res = 0;
    for(int i = 1;i <= 20210605;i++) {
        if(is_pure(i) && is_prime(i))
            ++res;
    }
    cout << res << '\n';
    return 0;
}