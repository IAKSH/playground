// https://sim.csp.thusaac.com/contest/35/problem/1
// 16AC 4TLE

#include <bits/stdc++.h>
using namespace std;

int main() {
    string s,tmp;
    getline(cin,s);
    s = s.substr(1,s.size() - 2);  // 去掉首尾的引号

    unordered_map<char,char> trans;

    stringstream ss;
    int n,m,k;
    getline(cin,tmp);
    ss << tmp;
    ss >> n;

    for(int i = 0;i < n;i++) {
        getline(cin,tmp);
        trans[tmp[1]] = tmp[2];
    }

    cin >> m;
    for(int i = 0;i < m;i++) {
        cin >> k;
        string trans_s = s;

        // 预计算每个字符在 k 次转换后的结果
        unordered_map<char,char> final_trans;
        for(auto& p : trans) {
            char current = p.first;
            char next = p.second;
            for(int j = 1;j < k;j++) {
                if(trans.count(next) > 0) {
                    next = trans[next];
                }
                else
                    break;
            }
            final_trans[current] = next;
        }

        for (auto& c : trans_s) {
            if (final_trans.count(c)) {
                c = final_trans[c];
            }
        }
        cout << '#' << trans_s << "#\n";
    }

    return 0;
}
