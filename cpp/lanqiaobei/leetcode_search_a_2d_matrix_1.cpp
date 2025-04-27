// https://leetcode.cn/problems/search-a-2d-matrix/description/?envType=study-plan-v2&envId=top-100-liked
// 也许是二分
// 不知道为什么会被leetcode报越界

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int w = matrix[0].size(), h = matrix.size();
        vector<int> v(w * h);
        for(int i = 0;i < h;i++) {
            for(int j = 0;j < w;j++) {
                v[i * (h + 1) + j] = matrix[i][j];
            }
        }
        auto it = lower_bound(v.begin(),v.end(),target);
        return (it != v.end() && *it == target);
    }
};

int main() {
    //vector<vector<int>> matrix = {{1,3,5,7},{10,11,16,20},{23,30,34,60}};
    //vector<vector<int>> matrix = {{1}};
    vector<vector<int>> matrix = {{1,3}};
    std::cout << (Solution().searchMatrix(matrix,1) ? "true" : "false") << '\n';
    std::cout << (Solution().searchMatrix(matrix,3) ? "true" : "false") << '\n';
    std::cout << (Solution().searchMatrix(matrix,0) ? "true" : "false") << '\n';
    std::cout << (Solution().searchMatrix(matrix,4) ? "true" : "false") << '\n';
    return 0;
}