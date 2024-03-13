// https://www.dotcpp.com/oj/problem3155.html

#include <bits/stdc++.h>

using namespace std;

/**
 * 需要用小顶堆来优化时间复杂度
 * 但在这之前，需要搞明白为什么结果错误
 * https://www.dotcpp.com/oj/submit_status.php?sid=15757638
*/

int main() noexcept {
    int n,k;
    cin >> n >> k;

    list<int> nums(n);
    for(auto& i : nums) {
        cin >> i;
    }

    for(int i = 0;i < k;i++) {
        auto min_it = min_element(nums.begin(),nums.end());
        auto it = min_it;
        if(++it != nums.end()) {
            auto right_it = it;
            *right_it += *(--it);
        }
        it = min_it;
        if(it != nums.begin()) {
            auto left_it = --it;
            *left_it += *(++it);
        }
        nums.erase(min_it);
    }

    for(const auto& i : nums) {
        cout << i << ' ';
    }
    cout << '\n';
    return 0;
}