// https://leetcode.cn/problems/spiral-matrix/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int x = 0,y = 0,nx,ny,cnt = 0;
        int dir = 0;
        int total = matrix.size() * matrix[0].size();

        int offset[][2] {
            {1,0},
            {0,1},
            {-1,0},
            {0,-1}
        };

        vector<int> results(total);

        while(cnt != total) {
            results[cnt++] = matrix[y][x];
            matrix[y][x] = INT_MIN;
            nx = x + offset[dir][0];
            ny = y + offset[dir][1];
            if(nx < 0 || nx >= matrix[0].size() || ny < 0 || ny >= matrix.size() || matrix[ny][nx] == INT_MIN) {
                dir = (dir + 1) % 4;
                nx = x + offset[dir][0];
                ny = y + offset[dir][1];
            }
            x = nx;
            y = ny;
        }

        return results;
    }
};

int main() {
    vector<vector<int>> matrix{
        {1,2,3},
        {4,5,6},
        {7,8,9}
    };

    for(const auto& i : Solution().spiralOrder(matrix))
        cout << i << ',';
    cout << "\b \n";

    return 0;
}

/*
输入：matrix = [[1,2,3],[4,5,6],[7,8,9]]
输出：[1,2,3,6,9,8,7,4,5]
*/