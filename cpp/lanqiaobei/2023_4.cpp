// https://www.dotcpp.com/oj/problem3153.html

#include <bits/stdc++.h>

using namespace std;

/**
 * https://www.dotcpp.com/oj/submit_status.php?sid=15736654
 * 运行时间: 9ms    消耗内存: 2092KB
*/

template <int BLOCKING,int NEW_VAL,bool EIGHT_WAY>
void dyeing(vector<vector<int>>& mat,int m,int n,int x,int y) noexcept {
    if(x > n + 1 || y > m + 1 || x < 0 || y < 0 || mat[x][y] == BLOCKING || mat[x][y] == NEW_VAL) {
        return;
    }

    mat[x][y] = NEW_VAL;

    dyeing<BLOCKING,NEW_VAL,EIGHT_WAY>(mat,m,n,x + 1,y);          // right
    dyeing<BLOCKING,NEW_VAL,EIGHT_WAY>(mat,m,n,x,y + 1);          // down
    dyeing<BLOCKING,NEW_VAL,EIGHT_WAY>(mat,m,n,x - 1,y);          // left
    dyeing<BLOCKING,NEW_VAL,EIGHT_WAY>(mat,m,n,x,y - 1);          // up
    // there's no constexpr in C++ 11 :(
    if(EIGHT_WAY) {
        dyeing<BLOCKING,NEW_VAL,EIGHT_WAY>(mat,m,n,x - 1,y - 1);  // left-up
        dyeing<BLOCKING,NEW_VAL,EIGHT_WAY>(mat,m,n,x - 1,y + 1);  // left-down
        dyeing<BLOCKING,NEW_VAL,EIGHT_WAY>(mat,m,n,x + 1,y - 1);  // right-up
        dyeing<BLOCKING,NEW_VAL,EIGHT_WAY>(mat,m,n,x + 1,y + 1);  // right-down
    }
}

int count_island(vector<vector<int>>& mat,int m,int n) noexcept {
    // 思路是阔图+两次染色（两次dfs）
    // 这两个dfs的时间和空间复杂度都比较高，但是好在测试集的mat大小应该都不太大
    for(int i = 0;i < m;i++) {
        string line;
        cin >> line;
        for(int j = 0;j < n;j++) {
            mat[j + 1][i + 1] = line[j] - '0'; 
        }
    }

    dyeing<1,-1,true>(mat,m,n,0,0);
    for(auto& l : mat) {
        for(auto& i : l) {
            i++;
        }
    }
    // 现在，所有大于0的都是陆地

    int cnt = 0;
    for(int i = 1;i < m + 1;i++) {
        for(int j = 1;j < n + 1;j++) {
            if(mat[j][i] > 0) {
                ++cnt;
                dyeing<0,0,false>(mat,m,n,j,i);
            }
        }
    }

    return cnt;
}

int main() noexcept {
    int t;
    cin >> t;
    vector<int> results(t);
    for(int i = 0;i < t;i++) {
        int m,n;
        cin >> m >> n;
        vector<vector<int>> mat(n + 2,vector<int>(m + 2));
        results[i] = count_island(mat,m,n);
    }

    for(const auto& i : results) {
        cout << i << '\n';
    }
    return 0;
}