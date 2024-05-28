// https://www.lanqiao.cn/problems/116/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B
// 就是快速幂，以及一点点数学吧
// 100%

#include <bits/stdc++.h>

#define USE_QUICK_POWER

using namespace std;

long long quick_power(long long a, long long b,long long c) {
	long long m = a,n = b,res = 1;
    while(n > 0) {
        if(n & 1)// 更快的n % 2 == 1
            res = res * m % c;
        m = m * m % c;
        n >>= 1;// 也许更快的n /= 2;
    }
    return res % c;
}

int main() {
    ios::sync_with_stdio(false);

    int a,b,n;
    cin >> a >> b >> n;

    // 因为求小数n位开始的三位，所以直接把a按10^(n+2)扩大，然后再取模，就能得到除出来结果中小数的n位开始的三位的被除数了
    // 因为同余定理，所以实际上可以这样写
    // b * 1000是因为要的是小数部分的n位后三位，b扩大1000倍来取余就可以保留最后三位了
    // 最后再把得出的这个被除数直接除b，就得到要的拿三位了
#ifdef USE_QUICK_POWER
    cout << (a * quick_power(10,n + 2,b * 1000) % (b * 1000)) / b << '\n';
#else
    cout << (a * static_cast<long long>(pow(10,n + 2)) % (b * 1000)) / b << '\n';
#endif
    return 0;
}