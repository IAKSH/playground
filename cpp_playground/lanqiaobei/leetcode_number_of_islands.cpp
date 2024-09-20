// https://leetcode.cn/problems/number-of-islands/description/?envType=study-plan-v2&envId=top-100-liked
// 由于内岛也被看做了岛，所以只需要DFS就行了

#include <iostream>
#include <vector>

using namespace std;

class Solution {
public:
    int numIslands(vector<vector<char>>& grid) {
        int cnt = 0;
        for(int i = 0;i < grid.size();i++) {
            for(int j = 0;j < grid[0].size();j++) {
                cnt += dfs(grid,j,i,0,0);
            }
        }
        return cnt;
    }

private:
    bool dfs(vector<vector<char>>& grid,int x,int y,int dx,int dy) {
        if(x >= 0 && y >= 0 && y < grid.size() && x < grid[0].size() && grid[y][x] == '1') {
            grid[y][x] = '0';

            if(dx != 1)
                dfs(grid,x - 1,y,-1,0);
            if(dx != -1)
                dfs(grid,x + 1,y,1,0);
            if(dy != 1)
                dfs(grid,x,y - 1,0,-1);
            if(dy != -1)
                dfs(grid,x,y + 1,0,1);

            return true;
        }

        return false;
    }
};

int main() {
    std::vector<std::vector<char>> grid = {
        {'1', '1', '1', '1', '0'},
        {'1', '1', '0', '1', '0'},
        {'1', '1', '0', '0', '0'},
        {'0', '0', '0', '0', '0'}
    };

    cout << Solution().numIslands(grid) << '\n';
    return 0;
}