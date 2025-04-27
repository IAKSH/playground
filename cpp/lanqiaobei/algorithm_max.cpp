#include <algorithm>
#include <iostream>
#include <vector>

// std::min的用法和std::max一样

using namespace std;

int main() {
    int a = 114;
    int b = 514;
    int c = 19;
    int d = 19;

    cout << max({a,b,c,d}) << '\n';
    cout << max('a','b') << '\n';

    // std::max不能直接从STL容器中获取最大值
    // 但是std::max_element可以

    return 0;
}
