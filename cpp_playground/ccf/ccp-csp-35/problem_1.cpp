#include <bits/stdc++.h>
// 16AC 4TLE

using namespace std;

int main() {
    string s,trans_s,tmp;
    getline(cin,s);
    s.erase(s.begin());
    s.erase(s.end() - 1);

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
        trans_s = s;
        for(int j = 0;j < k;j++) {
            for(auto& c : trans_s) {
                if(trans.count(c) > 0)
                    c = trans[c];
            }
        }
        cout << '#' << trans_s << "#\n";
    }

    return 0;
}
