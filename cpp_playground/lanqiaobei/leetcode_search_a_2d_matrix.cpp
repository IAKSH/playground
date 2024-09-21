// https://leetcode.cn/problems/search-a-2d-matrix/description/?envType=study-plan-v2&envId=top-100-liked
// 也许是二分

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int line_index = get_index(target,matrix.size(),[&](int i){return matrix[i].back();});
        if(line_index < matrix.size()) {
            auto& line = matrix[line_index];
            int col_index = get_index(target,line.size(),[&](int i){return line[i];});
            return col_index < line.size() && line[col_index] == target;
        }
        return false;
    }

private:
    int get_index(int target,int len,function<int(int)> get_val_callback) {
        int l = 0, r = len, m, m_val;
        while(l < r) {
            m = (l + r) / 2;
            m_val = get_val_callback(m);
            if(m_val > target)
                r = m;
            else if(m_val < target)
                l = m + 1;
            else
                return m;
        }
        return l;
    }
};

int main() {
    vector<vector<int>> matrix = {{1,3,5,7},{10,11,16,20},{23,30,34,60}};
    //vector<vector<int>> matrix = {{1}};
    //vector<vector<int>> matrix = {{1,3}};
    std::cout << (Solution().searchMatrix(matrix,3) ? "true" : "false") << '\n';
    return 0;
}