// https://www.dotcpp.com/oj/problem3152.html

// 疑似 O(N^2)遍历 * 线性DP

#include <algorithm>
#include <vector>
#include <cstdio>

using Ite = std::vector<int>::iterator;

int dp(Ite&& begin,Ite&& end) noexcept {
    
}

int main() noexcept {
    int n;
    std::vector<int> v(n);
    for(int i = 0;i < n;i++) {
        int input;
        std::scanf("%d",&input);
        v[i] = input;
    }

    std::vector<int> res;
    for(int i = 0;i < n;i++) {
        res.emplace_back(dp(std::begin(v) + i,std::end(v)));
    }
    printf("%d\n",std::min(std::begin(res),std::end(res)));
}