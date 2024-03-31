// https://www.lanqiao.cn/problems/158/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n;cin >> n;
    vector<int> v(n);
    for(auto& i : v)
        cin >> i;
    
    int max_len = INT_MIN;
    for(int i = 0;i < n;i++) {
        int len = 1;
        for(int j = i + 1;j < n;j++) {
            if(v[j] <= v[j - 1])
                break;
            ++len;
        }
        max_len = max(max_len,len);
    }

    cout << max_len << '\n';
    return 0;
}