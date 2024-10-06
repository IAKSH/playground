// https://leetcode.cn/problems/set-matrix-zeroes/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        array<vector<int>,2> arr{vector<int>(matrix.size()),vector<int>(matrix[0].size())};
        for(int i = 0;i < matrix.size();i++) {
            for(int j = 0;j < matrix[0].size();j++) {
                if(matrix[i][j] == 0) {
                    arr[0][i] = 1;
                    arr[1][j] = 1;
                }
            }
        }
        for(int i = 0;i < matrix.size() || i < matrix[0].size();i++) {
            if(i < matrix.size() && arr[0][i])
                fill(matrix[i].begin(),matrix[i].end(),0);
            if(i < matrix[0].size() && arr[1][i]) {
                for(int j = 0;j < matrix.size();j++) {
                    matrix[j][i] = 0;
                }
            }
        }
    }
};

void print_matrix(const vector<vector<int>>& matrix) {
    for(const auto& line : matrix) {
        for(const auto& col : line)
            cout << col << ' '; 
        cout << '\n';
    }
}

int main() {
    //[[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    vector<vector<int>> matrix{{0,1,2,0},{3,4,5,2},{1,3,1,5}};
    print_matrix(matrix);
    Solution().setZeroes(matrix);
    cout << "after:\n";
    print_matrix(matrix);
    return 0;
}