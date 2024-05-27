// https://www.lanqiao.cn/problems/1021/learning/?page=1&first_category_id=1&tags=%E5%9B%BD%E8%B5%9B,%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92&sort=pass_rate&asc=0
// 试图逃课，然后发现似乎并不能逃课，因为并不是连续上升

#include <bits/stdc++.h>

using namespace std;

const string input = "lanqiao";
//"tocyjkdzcieoiodfpbgcncsrjbhmugdnojjddhllnofawllbhf"\
//"iadgdcdjstemphmnjihecoapdjjrprrqnhgccevdarufmliqij"\
//"gihhfgdcmxvicfauachlifhafpdccfseflcdgjncadfclvfmad"\
//"vrnaaahahndsikzssoywakgnfjjaihtniptwoulxbaeqkqhfwl";

bool check(string::const_iterator it1,string::const_iterator it2) {
    if(it2 - it1 == 1)
        return true;
    char last = *it1;
    for(++it1;it1 != it2;it1++) {
        if(*it1 <= last)
            return false;
        last = *it1;
        
    }
    return true;
}

bool check(string s) {
    return check(s.begin(),s.end());
}

int main() {
    ios::sync_with_stdio(false);

    cout << (check("ln") ? "true\n" : "false\n");
    cout.flush();

    int cnt = 0,len = input.size();
    unordered_set<string> set;

    stringstream ss;
    for(int i = 1;i < len;i++) {
        for(int j = 0;j <= len - i;j++) {
            if(check(input.begin() + j,input.begin() + j + i)) {
                ss.str("");
                ss << input.substr(j,i);
                if(set.find(ss.str()) == set.end()) {
                    set.emplace(ss.str());
                    ++cnt;
                    cout << cnt << '\t' << ss.str() << '\n';
                }
            }
        }
    }

    cout << cnt << '\n';
    return 0;
}