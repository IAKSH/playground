// https://www.luogu.com.cn/problem/P5788
// 6AC 4LTE
// 似乎内存操作太多了，并没有比之前O(n^2)更快

#include <bits/stdc++.h>

using namespace std;

struct Data {
    int val;
    int index;
};

auto comp = [](const Data& m,const Data& n) {
    return m.val < n.val;
};

int main() {
    ios::sync_with_stdio(false);

    int n;cin >> n;
    stack<Data> s,s1;
    priority_queue<Data,std::deque<Data>,decltype(comp)> pq(comp);
    vector<int> res(n);
    for(int i = 0;i < n;i++) {
        Data d;
        cin >> d.val;
        d.index = i;
        pq.emplace(d);
    }

    while(!pq.empty()) {
        while(true) {
            if(s.empty()) {
                s.emplace(pq.top());
                res[pq.top().index] = 0;
                pq.pop();
                while(!s1.empty()) {
                    s.emplace(s1.top());
                    s1.pop();
                }
                break;
            }
            else if(pq.top().index > s.top().index || pq.top().val > s.top().val) {
                s1.emplace(s.top());
                s.pop();
            }
            else {
                res[pq.top().index] = s.top().index + 1;
                s.emplace(pq.top());
                pq.pop();
                while(!s1.empty()) {
                    s.emplace(s1.top());
                    s1.pop();
                }
                break;
            }
        }
    }

    for(const auto& i : res)
        cout << i << ' ';
    return 0;
}