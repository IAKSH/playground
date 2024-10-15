// https://leetcode.cn/problems/search-a-2d-matrix-ii/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        return dfs(matrix,target,0,matrix[0].size() - 1);
    }

private:
    bool dfs(const vector<vector<int>>& matrix, int target,int i,int j) {
        if(i >= matrix.size() || j < 0)
            return false;
        if(matrix[i][j] == target)
            return true;
        else if(matrix[i][j] > target)
            return dfs(matrix,target,i,j - 1);
        else
            return dfs(matrix,target,i + 1,j);
    }
};

int main() {
    vector<vector<int>> matrix{
        {1,4,7,11,15},
        {2,5,8,12,19},
        {3,6,9,16,22},
        {10,13,14,17,24},
        {18,21,23,26,30}
    };

    cout << (Solution().searchMatrix(matrix,5) ? "true" : "false") << '\n';

    return 0;
}