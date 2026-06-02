// https://leetcode.cn/problems/pascals-triangle/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

class Solution1 {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> m{{1}};
        for(int line = 1;line < numRows;line++) {
            m.emplace_back(vector<int>());
            for(int col = 0;col < line + 1;col++) {
                int l = 0, r = 0;
                if(col - 1 >= 0)
                    l = m[m.size() - 2][col - 1];
                if(col < m[m.size() - 2].size())
                    r = m[m.size() - 2][col];
                m[m.size() - 1].emplace_back(l + r);
            }
        }
        return m;
    }
};

class Solution {
public:
    vector<vector<int>> generate(int numRows) {
        vector<vector<int>> m(numRows);
        m[0].resize(1);
        m[0][0] = 1;
        int i,j,l,r;
        for(i = 1;i < numRows;i++) {
            m[i].resize(i + 1);
            for(j = 0;j < i + 1;j++) {
                l = r = 0;
                if(j != 0)
                    l = m[i - 1][j - 1];
                if(j != i)
                    r = m[i - 1][j];
                m[i][j] = l + r;
            }
        }
        return m;
    }
};

int main() {

}