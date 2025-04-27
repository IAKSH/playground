#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int climbStairs(int n) noexcept {
        int first = 1;
        if(n == 1) {
            return first;
        }

        int second = 2;
        if(n == 2) {
            return second;
        }

        int last;
        for(int i = 2;i < n;i++) {
            last = second + first;
            first = second;
            second = last;
        }
        return last;
    }
};

int main() noexcept {
    Solution s;
    cout << s.climbStairs(44) << '\n';
    return 0;
}