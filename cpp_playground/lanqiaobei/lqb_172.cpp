// https://www.lanqiao.cn/problems/172/learning/?page=1&first_category_id=1&second_category_id=3&problem_id=172
#ifdef OLD_VER

#include <bits/stdc++.h>

using namespace std;

/**
 * 暴力搜索
 * 两例超时
*/
int main() noexcept {
    int n;
    cin >> n;
    vector<vector<int>> data(3,vector<int>(n));
    for(auto& v : data) {
        for(auto& val : v) {
            cin >> val;
        }
    }

    for(auto& v : data) {
        sort(v.begin(),v.end());
    }

    int cnt = 0;
    for(int i = 0;i < n;i++) {
        for(int j = 0;j < n;j++) {
            if(data[1][j] <= data[0][i])
                continue;
            for(int k = 0;k < n;k++) {
                if(data[2][k] <= data[1][j])
                    continue;
                cnt += n - k;
                break;
            }
        }
    }

    cout << cnt << '\n';
    return 0;
}
#else

#include <bits/stdc++.h>

using namespace std;

/**
 * 二分查找
 * 1例超时，1例错误
*/

int main() noexcept {
    int n;
    cin >> n;
    vector<vector<int>> data(3,vector<int>(n));
    for(auto& v : data) {
        for(auto& val : v) {
            cin >> val;
        }
    }

    for(auto& v : data) {
        sort(v.begin(),v.end());
    }

    int cnt = 0;
    for(int i = 0;i < n;i++) {
        int j = lower_bound(data[1].begin(),data[1].end(),data[0][i] + 1) - data[1].begin();
        for(;j < n;j++) {
            int k = lower_bound(data[2].begin(),data[2].end(),data[1][j] + 1) - data[2].begin();
            cnt += n - k;
        }
    }

    cout << cnt << '\n';
    return 0;
}
#endif