// https://sim.csp.thusaac.com/contest/35/problem/3
// WA & TLE

#include <bits/stdc++.h>

using namespace std;

struct Node {
    int x,y;
    bool mark = false;
    unordered_map<Node*,int> nearby;
};

struct Station {
    int x,y,r,t;
};

int n,m,a,b,c,d;
vector<Node> nodes;
vector<Station> stations;
int min_result = INT_MAX;

void dfs(Node* node,int t) {
    if(node == &nodes.back() && t < min_result)
        min_result = t;
    else {
        for(auto& other : node->nearby) {
            if(!other.first->mark) {
                other.first->mark = true;
                dfs(other.first,t + other.second);
            }
        }
    }
    node->mark = false;
}

int main() {
    cin >> n >> m;

    for(int i = 0;i < n;i++) {
        cin >> a >> b;
        nodes.emplace_back(Node{a,b});
    }

    for(int i = 0;i < m;i++) {
        cin >> a >> b >> c >> d;
        stations.emplace_back(Station{a,b,c,d});
    }

    // 求第一个节点到最后一个节点的最小延迟
    // 根据距离建图，然后图中dfs（由于距离不等，不用bfs）

    for(const auto& station : stations) {
        vector<Node*> nearby_nodes;
        for(auto& node : nodes) {
            if(abs(node.x - station.x) <= station.r && abs(node.y - station.y) <= station.r)
                nearby_nodes.emplace_back(&node);
        }
        for(auto& node : nearby_nodes) {
            for(const auto& other : nearby_nodes) {
                if(other != node)
                    node->nearby[other] = min(node->nearby[other] == 0 ? INT_MAX : node->nearby[other],station.t);
            }
        }
    }

    dfs(&nodes.front(),0);
    cout << min_result << '\n';
    return 0;
}