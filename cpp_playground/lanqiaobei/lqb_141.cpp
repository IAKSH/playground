// https://www.lanqiao.cn/problems/141/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=20

#include <bits/stdc++.h>

using namespace std;

pair<int,int> foundA(int n,const vector<vector<char>>& mat) noexcept {
    for(int i = 0;i < n;i++) {
        for(int j = 0;j < n;j++) {
            if(mat[i][j] == 'A') {
                return pair<int,int>{i,j};
            }
        }
    }
    return pair<int,int>{-1,-1};
}

struct Node {
    int i,j;
    bool b;
};

int main() noexcept {
    int n; cin >> n;
    vector<vector<char>> mat(n,vector<char>(n));
    for(auto& line : mat) {
        for(auto& c : line) {
            cin >> c;
        }
    }

    queue<Node> bfs_queue;
    auto a = foundA(n,mat);
    if(a.first + 1 < n)
        bfs_queue.emplace(Node{a.first + 1,a.second,(mat[a.first + 1][a.second] == '+' ? true : false)});
    if(a.second + 1 < n)
        bfs_queue.emplace(Node{a.first,a.second + 1,(mat[a.first][a.second + 1] == '+' ? true : false)});
    if(a.first - 1 >= 0)
        bfs_queue.emplace(Node{a.first - 1,a.second,(mat[a.first - 1][a.second] == '+' ? true : false)});
    if(a.second - 1 >= 0)
        bfs_queue.emplace(Node{a.first,a.second - 1,(mat[a.first][a.second - 1] == '+' ? true : false)});

    int cnt = 1;
    while(!bfs_queue.empty()) {
        int len = bfs_queue.size();
        for(int _i = 0;_i < len;_i++) {
            auto pos = bfs_queue.front();
            if(mat[pos.i][pos.j] == (pos.b ? '+' : '-')) {
                //cout << pos.i << ',' << pos.j << '\n';
                mat[pos.i][pos.j] = '0';
                if(pos.i + 1 < n)
                    bfs_queue.emplace(Node{pos.i + 1,pos.j,!pos.b});
                if(pos.j + 1 < n)
                    bfs_queue.emplace(Node{pos.i,pos.j + 1,!pos.b});
                if(pos.i - 1 >= 0)
                    bfs_queue.emplace(Node{pos.i - 1,pos.j,!pos.b});
                if(pos.j - 1 >= 0)
                    bfs_queue.emplace(Node{pos.i,pos.j - 1,!pos.b});
            }
            else if(mat[pos.i][pos.j] == 'B') {
                cout << cnt << '\n';
                return 0;
            }
            bfs_queue.pop();
        }
        ++cnt;
        //cout << "---\n";
    }
    
    cout << -1 << '\n';
    return 0;
}