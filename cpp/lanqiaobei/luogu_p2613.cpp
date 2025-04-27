// https://www.luogu.com.cn/problem/P2613
// 似乎得上高精
// 目前做的纯粹是数学方案
// nmd,wsm
// 10WA

#include <bits/stdc++.h>

using namespace std;

const int MOD = 19260817;

// 从之前代码搬运过来的快速幂
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

    long long a,b;
    cin >> a >> b;

    if(b == 0)
        cout << "Angry!\n";
    else {
        long long tmp = a * quick_power(b,MOD - 2,MOD);
        cout << (tmp % MOD + MOD) % MOD << '\n';
    }
	
    return 0;
}