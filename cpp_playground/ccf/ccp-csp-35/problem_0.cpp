// https://sim.csp.thusaac.com/contest/35/problem/0
// AC

#include <bits/stdc++.h>

using namespace std;

int get_level(const string& s) {
    bool has_lower_alphabet = false;
    bool has_upper_alphabet = false;
    bool has_num = false;
    bool has_special = false;

    bool unique = true;
    unordered_map<char,int> mem;

    for(int i = 0;i < s.size();i++) {
        if(unique) {
            if(mem[s[i]] > 1)
                unique = false;
            else
                ++mem[s[i]];
        }
        if(!has_lower_alphabet && s[i] >= 'a' && s[i] <= 'z')
            has_lower_alphabet = true;
        if(!has_upper_alphabet && s[i] >= 'A' && s[i] <= 'Z')
            has_upper_alphabet = true;
        if(!has_num && s[i] >= '0' && s[i] <= '9')
            has_num = true;
        if(!has_special && (s[i] == '*' || s[i] == '#'))
            has_special = true;
    }

    if((has_lower_alphabet || has_upper_alphabet) && has_num && has_special && unique)
        return 2;
    else if((has_lower_alphabet || has_upper_alphabet) && has_num && has_special)
        return 1;
    else
        return 0;
}

int main() {
    int n;
    cin >> n;
    string s;
    for(int i = 0;i < n;i++) {
        cin >> s;
        cout << get_level(s) << '\n';
    }
    return 0;
}