// https://sim.csp.thusaac.com/contest/35/problem/2
// shit

#include <bits/stdc++.h>

using namespace std;

void run() {
    int n;
    string dummy;
    stringstream ss[4];

    cin >> n;
    getline(cin,dummy);

    deque<string> ori;
    for(int i = 0;i < n;i++) {
        ori.emplace_back(string());
        getline(cin,ori.back());
    }

    getline(cin,dummy);

    while(getline(cin,dummy),dummy != "") {
        int a,b,c,d;// 原文件起始，源文件涉及行数，新文件起始，新文件行数
                    // 关于检测由这几个值不匹配造成的包损坏还没有实现
        while(dummy[0] == '@')
            dummy.erase(0,1);
        sscanf(dummy.data()," -%d,%d +%d,%d @@",&a,&b,&c,&d);
        
        bool flag = true;
        int i = a;
        while(flag) {
            getline(cin,dummy);
            switch(dummy[0]) {
                case '-':
                    dummy.erase(0,1);
                    if(dummy == ori[i]) {
                        ori.erase(ori.begin() + i);
                        --i;
                    }
                    else
                        throw runtime_error("1");
                    break;
                case '+':
                    ++i;
                    dummy.erase(0,1);
                    ori.insert(ori.begin() + i,dummy);
                    break;
                case ' ':
                    dummy.erase(0,1);
                    if(dummy != ori[i])
                        throw runtime_error("2");
                    ++i;
                    break;
                case '@':
                default:
                    flag = false;
            }
        }
    }

    for(const auto& s : ori)
        cout << s << '\n';
}

int main() {
    //try {
    //    run();
    //}
    //catch(const exception& e) {
    //    cout << e.what() << '\n';
    //    cout << "Patch is damaged.\n";
    //}
    run();
    return 0;
}