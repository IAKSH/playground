// https://sim.csp.thusaac.com/contest/35/problem/1

#include <bits/stdc++.h>

using namespace std;

int main() {
    string s,temp_s,input;
    getline(cin,s);
    s.erase(s.begin());
    s.erase(s.end() - 1);

    unordered_map<char,char> trans,tmp_map_1,tmp_map_2;
    stringstream ss;

    int n,m,k;
    getline(cin,input);
    ss << input;
    ss >> n;

    for(int i = 0;i < n;i++) {
        getline(cin,input);
        trans[input[1]] = input[2];
    }

    cin >> m;
    for(int i = 0;i < m;i++) {
        cin >> k;
        temp_s = s;
        tmp_map_1 = trans;
        tmp_map_2 = trans;

        for(int j = 0;j < k - 1;j++) {
            for(auto& [k,v] : tmp_map_2) {
                char mk = v;
                bool b = true;
                while(tmp_map_1.count(mk) != 0) {
                    mk = tmp_map_1[mk];
                    if(mk == k) {
                        tmp_map_2.erase(k);
                        b = false;
                        break;
                    }
                }
                if(b)
                    v = mk;
            }
            tmp_map_1 = tmp_map_2;
        }

        for(auto& c : temp_s) {
            if(tmp_map_1.count(c) > 0)
                c = tmp_map_1[c];
        }
        cout << '#' << temp_s << "#\n";
    }

    return 0;
}