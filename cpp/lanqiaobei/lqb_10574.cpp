// https://www.lanqiao.cn/problems/10574/learning/?page=1

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,a;
    cin >> n;

    vector<int> v(n);
    for(auto& i : v) {
        scanf("%d",&i);
    }

    sort(v.begin(),v.end());
    for(int i = 0;i < v.size() - 1;i++) {
        if(v[i] + 1 < v[i + 1]) {
            cout << v[i] + 1 << '\n';
            return 0;
        }
    }
    cout << v.back() + 1 << '\n';
    return 0;
}