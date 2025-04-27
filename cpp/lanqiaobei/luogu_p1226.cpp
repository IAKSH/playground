// https://www.luogu.com.cn/problem/P1226
// 快速幂
// AC

#include <bits/stdc++.h>

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
    long long a,b,p;
    cin >> a >> b >> p;
    cout << a << '^' << b << " mod " << p << "=" << quick_power(a,b,p) << '\n';
    return 0;
}