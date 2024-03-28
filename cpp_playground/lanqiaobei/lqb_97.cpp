// https://www.lanqiao.cn/problems/97/learning/?page=1&first_category_id=1&second_category_id=3&tags=2017

#include <bits/stdc++.h>

using namespace std;

/**
 * 5个超时
 * 想不出来怎么优化，暂时这样吧
*/

int main() noexcept {
    int n,k;
    cin >> n >> k;

    vector<int> v(n);
    vector<int> prefix(n);
    for(int i = 0;i < n;i++) {
        cin >> v[i];
        //prefix[i] = (i > 0 ? prefix[i - 1] + v[i] : 0);
        prefix[i] = v[i] + (i > 0 ? prefix[i - 1] : 0);
    }

    int cnt = 0;
    for(int i = 0;i < n;i++) {
        for(int j = i;j < n;j++) {
            int sum = prefix[j] - (i > 0 ? prefix[i - 1] : 0);
            cnt += (sum % k == 0);
        }
    }

    cout << cnt << '\n';
    return 0;
}