// https://www.lanqiao.cn/problems/2412/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023

#include <bits/stdc++.h>

using namespace std;

#ifdef OLD_VER
int main() noexcept {
    int w,h,n,r;cin >> w >> h >> n >> r;
    vector<vector<bool>> mat(h,vector<bool>(w,false));
    for(int i = 0;i < n;i++) {
        int x,y;
        cin >> x >> y;
        for(int j = y - r;j <= y + r;j++) {
            if(j < 0)
                continue;
            if(j > h)
                break;
            int d = j - y;
            int l = sqrt(r * r - d * d);
            for(int k = x - l;k <= x + l;k++) {
                if(k < 0)
                    continue;
                if(k > w)
                    break;
                mat[j][k] = true;
            }
        }
    }
    int cnt = 0;
    for(const auto& line : mat) {
        for(const auto& b : line)
            cnt += !b;
    }
    cout << cnt << '\n';
    return 0;
}
#else
int distanceSquare(int x1,int y1,int x2,int y2) noexcept {
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

int main() noexcept {
    int w,h,n,r;cin >> w >> h >> n >> r;
    vector<pair<int,int>> v(n);
    for(auto& p : v)
        cin >> p.first >> p.second;

    int cnt = 0;
    for(int i = 0;i <= h;i++) {
        for(int j = 0;j <= w;j++) {
            for(const auto& p : v) {
                if(distanceSquare(p.first,p.second,j,i) <= r * r) {
                    ++cnt;
                    break;
                }
            }
        }
    }

    cout << cnt << '\n';
    return 0;
}
#endif

/*
10 10 2 5
0 0
7 0
=57
*/