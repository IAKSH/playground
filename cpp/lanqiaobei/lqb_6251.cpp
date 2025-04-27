// https://www.lanqiao.cn/problems/6251/learning/?first_category_id=1&page=1&second_category_id=3&difficulty=20&tags=2023
// 85% 3超时
// 笑死，结果是float精度不够，除法的时候精度丢失了
// 换double就好了
// 得出结论，不要在比赛里用float，数大一点就会有明显的精度丢失
// 还有一点，minmax_element没有下面这种自己写的O(n)的方法快，尽管前者理论上也是O(n)
// 我认为可能是minmax_element进行了多余的迭代器检查

#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n,k;cin >> n >> k;
    vector<int> v(n);
    for(auto& i : v) cin >> i;

    double sum = 0;
    int cnt = 0;// 也许有办法直接计算出cnt
    for(int i = 0;i <= n - k;i++) {
        int minn = INT_MAX;
        int maxn = INT_MIN;
        for(int j = i;j < i + k;j++) {
            if(v[j] > maxn)
                maxn = v[j];
            if(v[j] < minn)
                minn = v[j];
        } 
        sum += maxn - minn;
        ++cnt;
    }
    printf("%.2f\n",sum / cnt);
    return 0;
}

/*
3 2
1 2 3
=1.00

71 32
597948 86497 707587 208436 54917 331061 364228 582695 668447 29956 311133 670158 519914 501120 41467 249552 329287 354623 611231 352315 105710 166530 666905 482513 358786 193279 30155 576068 7726 535364 417758 469039 158401 279150 349379 538753 385044 672690 491943 537246 354761 15878 89193 350475 271911 34090 30732 609404 289449 196992 347422 238899 370917 251937 636800 316365 14939 656713 354273 331828 388258 245743 541932 89149 103677 40218 662002 496568 363396 708587 607892 
=667202.65
!=667202.88
*/