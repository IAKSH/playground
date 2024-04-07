// https://www.lanqiao.cn/problems/3491/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023

#include <bits/stdc++.h>

using namespace std;


int main() noexcept {
    vector<int> res;
    int n,len,a,b,cnt = 0;
    for(int i = 10;i < 100000000;i++) {
        res.clear();
        n = i;
        while(n > 0) {
            res.emplace_back(n % 10);
            n /= 10;
        }
        len = res.size();
        if(len % 2 == 0) {
            a = accumulate(res.begin(),res.begin() + len / 2,0);
            b = accumulate(res.begin() + len / 2,res.end(),0);
            cnt += (a == b);
        }
    }
    cout << cnt << '\n';
    return 0;
}