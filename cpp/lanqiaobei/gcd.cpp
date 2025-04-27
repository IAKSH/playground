#include <bits/stdc++.h>

using namespace std;

int gcd(int m,int n) noexcept {
    // 辗转相除法
#ifdef RECURSION
    int maxn = max(m,n);
    int minn = min(m,n);
    int r = maxn % minn;
    if(r != 0)
        return gcd(minn,r);
    else
        return minn;
#else
    int maxn = max(m,n);
    int minn = min(m,n);
    int r = maxn % minn;
    while(r != 0) {
        maxn = max(n,r);
        minn = min(n,r);
        r = maxn % minn;
    }
    return minn;
#endif
}

int main() noexcept {
    int m,n;
    cin >> m >> n;
    cout << gcd(m,n) << '\n';
    return 0;
}