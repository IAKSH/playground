// https://sim.csp.thusaac.com/contest/35/problem/3
// 10AC 2WA 8TLE

#include <bits/stdc++.h>

using namespace std;

struct Node {
    int x, y;
    unordered_map<Node*, int> nearby;
};

struct Station {
    int x, y, r, t;
};

int n, m, a, b, c, d;
vector<Node> nodes;
vector<Station> stations;
int min_result = INT_MAX;

int dijkstra(Node* start, Node* end) {
    unordered_map<Node*, int> distances;
    for(auto& node : nodes) {
        distances[&node] = INT_MAX;
    }
    distances[start] = 0;

    priority_queue<pair<int, Node*>, vector<pair<int, Node*>>, greater<pair<int, Node*>>> pq;
    pq.push({0, start});

    while(!pq.empty()) {
        auto [current_dist, current_node] = pq.top();
        pq.pop();

        if(current_node == end) {
            return current_dist;
        }

        for(auto& neighbor : current_node->nearby) {
            Node* next_node = neighbor.first;
            int weight = neighbor.second;
            int new_dist = current_dist + weight;

            if(new_dist < distances[next_node]) {
                distances[next_node] = new_dist;
                pq.push({new_dist, next_node});
            }
        }
    }

    return INT_MAX; // 若未找到路径返回无限大
}

int main() {
    cin >> n >> m;

    for(int i = 0; i < n; i++) {
        cin >> a >> b;
        nodes.emplace_back(Node{a, b});
    }

    for(int i = 0; i < m; i++) {
        cin >> a >> b >> c >> d;
        stations.emplace_back(Station{a, b, c, d});
    }

    for(const auto& station : stations) {
        vector<Node*> nearby_nodes;
        for(auto& node : nodes) {
            if(abs(node.x - station.x) <= station.r && abs(node.y - station.y) <= station.r)
                nearby_nodes.emplace_back(&node);
        }
        for(auto& node : nearby_nodes) {
            for(auto& other : nearby_nodes) {
                if(other != node)
                    node->nearby[other] = min(node->nearby[other] == 0 ? INT_MAX : node->nearby[other], station.t);
            }
        }
    }

    int result = dijkstra(&nodes.front(), &nodes.back());
    cout << (result == INT_MAX ? -1 : result) << '\n';
    return 0;
}
