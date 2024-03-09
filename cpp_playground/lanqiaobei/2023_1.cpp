// https://www.dotcpp.com/oj/problem3150.html
// 测试：3 75 3 53 2 59 2
// 输出：20 25

#include <algorithm>
#include <iostream>
#include <vector>
#include <climits>
#include <cmath>

int main()
{
    int n;
    int v_min = 0;
    int v_max = INT_MAX;

    std::cin >> n;

    for(int i = 0;i < n * 2;i += 2) {
        double a,b;
        std::cin >> a >> b;

        int v = std::floor(a / b);
        v_max = std::min(v,v_max);

        v = std::ceil(a / (b + 1));
        v_min = std::max(v,v_min);
    }
    
    std::cout << v_min << ' ' << v_max << '\n';
    return 0;
}