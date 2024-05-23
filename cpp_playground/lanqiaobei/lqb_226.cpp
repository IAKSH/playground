// https://www.lanqiao.cn/problems/226/learning/?page=1&first_category_id=1&tags=%E5%9B%BD%E8%B5%9B,%E8%B4%AA%E5%BF%83&sort=pass_rate&asc=0
// 就是贪心
// 看公式可以发现A的增长率比B大
// 所以应该是B优先用大的(来快速给A把d拉起来)，然后A最后用大的，在之前铺垫的基础上能够到达最大
// 100%

#include <bits/stdc++.h>

using namespace std;

int main() {
    int n,m,i,j;
    cin >> n >> m;

    vector<pair<long long,int>> a(n);// val,index + 1
    vector<pair<long long,int>> b(m);
    for(i = 0;i < n;i++) {
        cin >> a[i].first;
        a[i].second = i + 1;
    }
    for(i = 0;i < m;i++) {
        cin >> b[i].first;
        b[i].second = i + 1;
    }

    sort(a.begin(),a.end(),[](const pair<long long,int>& p1,const pair<long long,int>& p2){return p1.first < p2.first;});
    sort(b.begin(),b.end(),[](const pair<long long,int>& p1,const pair<long long,int>& p2){return p1.first > p2.first;});

    string s;
    cin >> s;

    i = j = 0;
    for(const auto& c : s) {
        if(c == '0') {
            cout << 'A' << a[i++].second << '\n';
            
        }
        else {
            cout << 'B' << b[j++].second << '\n';
        }
    }

    cout << "E\n";
    return 0;
}