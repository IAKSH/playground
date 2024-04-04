/**
 * https://www.lanqiao.cn/problems/3505/learning/?page=1&first_category_id=1&second_category_id=3&difficulty=10&tags=2023
 * 纯dfs会超时，尝试从dfs改造bfs
 * 为什么两种都只能过55%，bfs甚至有段错误
 * 懒得改了
*/

#include <bits/stdc++.h>

using namespace std;

static int n,m;
static vector<float> v;

#ifdef DFS
int min_cut_cnt = INT_MAX;

void dfs(float sum,int dn,int cut_cnt,int mark) noexcept {
    if(dn == n)
        return;
    
    switch(mark) {
    case 0:
        sum += v[dn];
        break;
    case 1:
        sum += v[dn] / 2.0f;
        ++cut_cnt;
        break;
    case 2:
        break;
    }

    if(sum > m)
        return;
    if(sum == m) {
        min_cut_cnt = min(min_cut_cnt,cut_cnt);
        return;
    }

    ++dn;
    for(int i = 0;i < 3;i++)
        dfs(sum,dn,cut_cnt,i);
}

int main() noexcept {
    cin >> n >> m;
    v.resize(n);
    for(auto& i : v)
        cin >> i;

    for(int i = 0;i < 3;i++)
        dfs(0,0,0,i);

    cout << min_cut_cnt << '\n';
    return 0;
}
#else
struct Node {
    float sum;
    int dn;
    int cut_cnt;
    int mark;
};

int main() noexcept {
    cin >> n >> m;
    v.resize(n);
    for(auto& i : v)
        cin >> i;

    queue<Node> nodes;
    for(int i = 0;i < 3;i++)
        nodes.push(Node{0,0,0,i});
    while(!nodes.empty()) {
        auto& node = nodes.front();
        // check node
        if(node.dn != n) {
            switch(node.mark) {
            case 0:
                node.sum += v[node.dn];
                break;
            case 1:
                node.sum += v[node.dn] / 2.0f;
                ++node.cut_cnt;
                break;
            default:
                break;
            }
            // if find the first one, output
            if(node.sum == m) {
                cout << node.cut_cnt << '\n';
                return 0;
            }
            // if not, add next nodes
            else if(node.sum < m) {
                for(int i = 0;i < 3;i++)
                    nodes.push(Node{node.sum,node.dn + 1,node.cut_cnt,i});
            }
        }
        nodes.pop();
    }
    
    return 1;
}
#endif