/**
 * https://www.lanqiao.cn/problems/3539/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023,%E8%B4%AA%E5%BF%83
 * 50% 2错误,3超时
*/

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n;cin >> n;
    vector<int> v(n);
    for(auto& i : v)
        cin >> i;

    sort(v.begin(),v.end());

    int sum = 0;
    for(int i = n - 1;i >= 0;i--) {
        if(v[i] == 0)
            continue;;
        for(int j = i - 1;j >= 0;j--) {
            if(v[j] == 0)
                continue;
            sum += v[i] + v[j];
            int half_min = min(v[i],v[j]) / 2;
            v[i] = 0;
            v[j] = 0;
            for(int k = n - 1;k >= 0;k--) {
                if(v[k] == 0)
                    continue;
                if(v[k] <= half_min) {
                    v[k] = 0;
                    break;
                }
            }
            break;
        }
    }

    cout << sum + v[0] << '\n';
    return 0;
}

/*
7
1 4 2 8 5 7 1
=25
*/