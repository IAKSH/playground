// https://www.lanqiao.cn/problems/2191/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B,2022
// 贪心
// 100%
// 最开始用的int，被爆了，换long long全过

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    
    long long n,m;
    cin >> n >> m;

    long long a[n],b[n];
    for(auto& i : a)
        cin >> i;
    for(auto& i : b)
        cin >> i;

    long long i,cnt = 0;
    bool should_exit,flag = false;
    while(!should_exit) {
        should_exit = false;
        if(m <= 0)
            break;
        for(i = 0;i < n;i++) {
            if(a[i] == 0 && b[i] == 0) {
                should_exit = true;
                break;
            }
        }

        flag = true;
        for(i = 0;i < n;i++) {
            if(a[i])
                --a[i];
            else if(m > 0 && b[i]) {
                --b[i];
                --m;
            }
            else {
                flag = false;
                break;
            }
        }
        if(flag)
            ++cnt;
    }

    cout << cnt << '\n';
    return 0;
}