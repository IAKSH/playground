// https://leetcode.cn/problems/search-a-2d-matrix/description/?envType=study-plan-v2&envId=top-100-liked
// 但是为什么暴力也能过

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        for(const auto& line : matrix) {
            for(const auto& col : line) {
                if(col == target)
                    return true;
            }
        }
        return false;
    }
};

int main() {
    vector<vector<int>> matrix = {{1,3,5,7},{10,11,16,20},{23,30,34,60}};
    std::cout << (Solution().searchMatrix(matrix,3) ? "true" : "false") << '\n';
    return 0;
}