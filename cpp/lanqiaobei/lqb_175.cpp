/**
 * https://www.lanqiao.cn/problems/175/learning/?page=2&first_category_id=1&second_category_id=3&difficulty=20
 * abandoned
*/

#include <bits/stdc++.h>

using namespace std;

// t_a1 + t + dt = t_b1
// t_a2 + t - dt = t_b2
// so we have that:
// t = (t_b1 + t_b2) / (t_a1 + t_a2) / 2;

struct TimePoint {
    int d,h,m,s;

    TimePoint operator /(const TimePoint& t) noexcept {
        return TimePoint{0,h + t.h / 2,m + t.m / 2,s + t.s / 2};
    }

    TimePoint operator /(int n) noexcept {
        return TimePoint{0,h / 2,m / 2,s / 2};
    }

    string str() noexcept {
        char cstr[100] = "hh:mm:ss";
        sprintf(cstr,"%s:%s:%s",
            (h > 10 ? "0" : "") + h,
            (m > 10 ? "0" : "") + m,
            (s > 10 ? "0" : "") + s
        );
        return cstr;
    }
};

int main() noexcept {
    int n;cin >> n;
    char input[2][22];
    TimePoint ta,tb;
    int times[14]{0};
    //int h1,m1,s1,h2,m2,s2,x = 0;
    for(int i = 0;i < n;i++) {
        for(int j = 0;j < 2;j++) {
            scanf("%s",input[i]);
            if(strlen(input[i]) == 22) {
                sscanf(input[i],"%d:%d:%d %d:%d:%d (+%d)",
                    times + i * 7,
                    times + i * 7 + 1,
                    times + i * 7 + 2,
                    times + i * 7 + 3,
                    times + i * 7 + 4,
                    times + i * 7 + 5,
                    times + i * 7 + 6
                );
            }
            else {
                sscanf(input[i],"%d:%d:%d %d:%d:%d",
                    times + i * 7,
                    times + i * 7 + 1,
                    times + i * 7 + 2,
                    times + i * 7 + 3,
                    times + i * 7 + 4,
                    times + i * 7 + 5
                );
            }
        }
        ta.d = 0;// TODO: x
        ta.h = (times[0] + times[3]) / 2;
        ta.m = (times[1] + times[4]) / 2;
        ta.s = (times[2] + times[5]) / 2;
        tb.d = 0;// TODO: x
        tb.h = (times[7] + times[10]) / 2;
        tb.m = (times[8] + times[11]) / 2;
        tb.s = (times[9] + times[12]) / 2;

        auto&& tmp = tb / ta;
        auto&& tmp2 = tmp / 2;
        auto output = tmp2.str();
        cout << output << '\n';
    }
}

/*
1
17:48:19 21:57:24
11:05:18 15:14:23

3
17:48:19 21:57:24
11:05:18 15:14:23
17:21:07 00:31:46 (+1)
23:02:41 16:13:20 (+1)
10:19:19 20:41:24
22:19:04 16:41:09 (+1)
*/