// https://leetcode.cn/problems/rotting-oranges/description/?envType=study-plan-v2&envId=top-100-liked
// 实际上应该是BFS

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        int cnt = -1;
        queue<pair<int,int>> que1,que2; 

        bool is_grid_zero = true;

        for(int i = 0;i < grid.size();i++) {
            for(int j = 0;j < grid[0].size();j++) {
                if(grid[i][j] != 0) {
                    is_grid_zero = false;
                    if(grid[i][j] == 2)
                        que2.push(make_pair(i,j));
                }
            }
        }

        if(is_grid_zero)
            return 0;

        while(!que2.empty()) {
                que1 = que2;
                while(!que2.empty())
                    que2.pop();

                while(!que1.empty()) {
                auto& p = que1.front();
                if(p.first > 0 && grid[p.first - 1][p.second] == 1) {
                    grid[p.first - 1][p.second] = 2;
                    que2.push(make_pair(p.first - 1,p.second));
                }
                if(p.first + 1 < grid.size() && grid[p.first + 1][p.second] == 1) {
                    grid[p.first + 1][p.second] = 2;
                    que2.push(make_pair(p.first + 1,p.second));
                }
                if(p.second > 0 && grid[p.first][p.second - 1] == 1) {
                    grid[p.first][p.second - 1] = 2;
                    que2.push(make_pair(p.first,p.second - 1));
                }
                if(p.second + 1 < grid[0].size() && grid[p.first][p.second + 1] == 1) {
                    grid[p.first][p.second + 1] = 2;
                    que2.push(make_pair(p.first,p.second + 1));
                }
                que1.pop();
            }

            ++cnt;
        }

        for(const auto& line : grid) {
            for(const auto& col : line) {
                if(col == 1) {
                    return -1;
                }
            }
        }

        return cnt;
    }
};

int main() {
    vector<vector<int>> grid = {
        {2,1,1},
        {1,1,0},
        {0,1,1}
    };

    //vector<vector<int>> grid = {{0}};

    std::cout << Solution().orangesRotting(grid) << '\n';
    return 0;
}