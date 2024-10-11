// https://leetcode.cn/problems/word-search/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        vector<vector<bool>> mark(board.size(),vector<bool>(board[0].size()));
        for(int i = 0;i < board.size();i++) {
            for(int j = 0;j < board[0].size();j++)
                if(dfs(board,mark,word,i,j,0))
                    return true;
        }
        return false;
    }

private:
    bool dfs(const vector<vector<char>>& board,vector<vector<bool>>& mark,const string& word,int i,int j,int cnt) {
        if(!mark[i][j] && cnt < word.size() && board[i][j] == word[cnt]) {
            mark[i][j] = true;
            if(cnt == word.size() - 1 ||
                ((i > 0 && dfs(board,mark,word,i - 1,j,cnt + 1)) ||
                (i < board.size() - 1 && dfs(board,mark,word,i + 1,j,cnt + 1)) ||
                (j > 0 && dfs(board,mark,word,i,j - 1,cnt + 1)) ||
                (j < board[0].size() - 1 && dfs(board,mark,word,i,j + 1,cnt + 1))))
                return true;
            mark[i][j] = false;
        }
        return false;
    }
};

int main() {
    vector<vector<char>> board{
        //{'A','B','C','E'},
        //{'S','F','C','S'},
        //{'A','D','E','E'}
        {'A','B'}
    };
    //cout << (Solution().exist(board,"ABCCED") ? "true" : "false") << '\n';
    cout << (Solution().exist(board,"BA") ? "true" : "false") << '\n';
    return 0;
}