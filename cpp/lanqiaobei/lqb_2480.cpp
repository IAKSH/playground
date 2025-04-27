/**
 * https://www.lanqiao.cn/problems/2480/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=30
 * abandoned
*/

#include <bits/stdc++.h>

using namespace std;

/*
8 2
8 1 2 3 9 4 7 10
10001010

8 3
8 1 2 3 9 4 7 10
10001010
*/

int main() noexcept {
    int n,k;
    cin >> n >> k;
    vector<int> v(n);
    vector<int> s;
    for(int i = 0;i < n;i++) {
        cin >> v[i];
    }
    int s_size = 0;
    string input;
    cin >> input;
    for(int i = 0;i < n;i++) {
        if(input[i] == '1') {
            s.emplace_back(v[i]);
            ++s_size;
        }
    }
    // dp[i][j] = (s[j] > s[j - 1] ? dp[i][j - 1] : dp[i][j - 1] + 1);
    vector<vector<int>> dp(s_size,vector<int>(s_size,0));
    for(int i = 0;i < s_size;i++) {
        int maxn = INT_MIN;
        for(int j = i;j < s_size;j++) {
            if(s[j] > maxn) {
                if(j > i)
                    dp[i][j] = dp[i][j - 1] + 1;
                else
                    dp[i][j] = 1;
                maxn = s[j];
            }
            else
                dp[i][j] = dp[i][j - 1];
        }
    }

    // const __gnu_cxx::__normal_iterator<std::vector<int>*, std::vector<std::vector<int> > >
    cout << max(dp.begin(),dp.end() - 1,[](const std::vector<std::vector<int>>::iterator& m,const std::vector<std::vector<int>>::iterator& n){
        return m->back() < n->back();
    })->back() + 1 << '\n';

    return 0;
}