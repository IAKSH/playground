// https://leetcode.cn/problems/rotate-image/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        for(int i = 0;i < matrix.size();i++) {
            for(int j = 0;j < matrix[0].size() - i;j++)
                swap(matrix[j][i],matrix[matrix.size() - 1 - i][matrix[0].size() - 1 - j]);
        }
        reverse(matrix.begin(),matrix.end());
    }
};

int main() {
    vector<vector<int>> matrix{{1,2,3},{4,5,6},{7,8,9}};
    Solution().rotate(matrix);
    for(const auto& v : matrix) {
        for(const auto& i : v)
            cout << i << ',';
        cout << "\b \n";
    }
    return 0;
}