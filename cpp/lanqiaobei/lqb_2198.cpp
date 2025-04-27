// https://www.lanqiao.cn/problems/2198/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B,2022&sort=pass_rate&asc=0
// 只能过一个，什么过拟合
// 16.7%
// 懒得修了，下次再说

#include <bits/stdc++.h>

using namespace std;

// 似乎就是（无向）（有环）图的遍历...不考虑优化的话
// 要是有环怎么办？BFS吗
// 不对，BFS没有优势，因为这里的路径长度还要考虑节点权重
// 还不如带环检测的DFS，检测到环则回退

struct Node {
    vector<shared_ptr<Node>> link;
    int val;
    Node(const int& val) : val(val) {}
};

//unordered_map<int,Node> map;
vector<std::shared_ptr<Node>> nodes;

unordered_set<int> memory;
// return len
int dfs(std::shared_ptr<Node> node,int aim,int len) {
    if(node->val == aim) {
        memory.erase(node->val);
        return len;
    }
    else {
        int weight = node->link.size();
        vector<int> res;
        for(const auto& i : node->link) {
            if(memory.find(i->val) == memory.end()) {
                memory.emplace(i->val);
                res.emplace_back(dfs(i,aim,len + weight));
            }
        }
        memory.erase(node->val);
        if(res.size())
            return *min_element(res.begin(),res.end());
        else
            return INT_MAX;
    }
}

int main() noexcept {
    int n,m,a,b;
    cin >> n >> m;
    // input nodes
    for(int i = 0;i < n;i++)
        nodes.emplace_back(make_shared<Node>(i));
    for(int i = 0;i < n - 1;i++) {
        cin >> a >> b;
        nodes[a - 1]->link.emplace_back(nodes[b - 1]);
        nodes[b - 1]->link.emplace_back(nodes[a - 1]);
    }
    // dfs
    vector<int> res;
    for(int i = 0;i < m;i++) {
        cin >> a >> b;
        // from nodes[a] dfs to nodes[b]
        // discard circle
        // select min len
        res.push_back(dfs(nodes[a - 1],b - 1,1));
    }
    // output
    for(const auto& i : res)
        cout << i << '\n';
    return 0;
}

/*
4 3
1 2
1 3
2 4
2 3
3 4
3 3

5
6
1
*/