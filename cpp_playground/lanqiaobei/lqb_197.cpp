// https://www.lanqiao.cn/problems/197/learning/?page=2&first_category_id=1&second_category_id=3&difficulty=20

#ifdef OLD_VER
#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,m;cin >> n >> m;
    vector<vector<int>> mat(n,vector<int>(m));
    for(auto& line : mat)
        for(auto& i : line)
            cin >> i;

    vector<vector<int>> trans_mat(m,vector<int>(n));
    for(int i = 0;i < n;i++)
        //for(int j = m - 1;j >= 0;j--)
        for(int j = 0;j < m;j++)
            trans_mat[j][n - i - 1] = mat[i][j];
    
    for(int i = 0;i < m;i++) {
        for(int j = 0;j < n;j++)
            cout << trans_mat[i][j] << ' ';
        cout << '\n';
    }

    return 0;
}
#else
#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,m;cin >> n >> m;
    vector<vector<int>> mat(n,vector<int>(m));
    for(auto& line : mat)
        for(auto& i : line)
            cin >> i;
    
    for(int i = 0;i < m;i++) {
        for(int j = 0;j < n;j++)
            cout << mat[n - j - 1][i] << ' ';
        cout << '\n';
    }

    return 0;
}
#endif