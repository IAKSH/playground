#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    long long res = 0;
    vector<long long> v(233333332);
    const int m_minus_2 = 2146516017;
    for(int i = 1;i <= 233333333;i++) {
        v[i] = pow(i,m_minus_2);
    }
    
    for(int i = 0;i < 233333332;i++) {
        for(int j = i;j < 233333332;j++) {
            res += (v[i] ^ v[j]);
        }
    }

    cout << res << '\n';
    return 0;
}