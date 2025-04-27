// https://www.lanqiao.cn/problems/2147/learning/?page=2&first_category_id=1&second_category_id=3&tags=2022

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,m;cin >> n >> m;
    vector<vector<int>> mat(n,vector<int>(m));
    for(auto& line : mat)
        for(auto & i : line)
            cin >> i;
    int limit;cin >> limit;
    
    int max_result = INT_MIN;
    for(int i = 2;i <= n;i++) {
        for(int j = 2;j <= m;j++) {
            for(int y = 0;y <= n - i;y++) {
                for(int x = 0;x <= m - j;x++) {
                    int maxn = INT_MIN;
                    int minn = INT_MAX;
                    for(int dy = y;dy < y + i;dy++) {
                        for(int dx = x;dx < x + j;dx++) {
                            maxn = max(maxn,mat[dy][dx]);
                            minn = min(minn,mat[dy][dx]);
                        }
                    }
                    if(maxn - minn <= limit) {
                        max_result = max(max_result,i * j);
                    }
                }
            }
        } 
    }
    cout << max_result << '\n';
    return 0;
}