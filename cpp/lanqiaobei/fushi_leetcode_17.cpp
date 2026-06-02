// https://leetcode.cn/problems/climbing-stairs/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>
using namespace std;

class Solution1 {
public:
    // dfs
    // TLE
    int climbStairs(int n) {
        if(n == 0)
            return 1;
        if(n < 0)
            return 0;
        return climbStairs(n - 1) + climbStairs(n - 2);
    }
};

class Solution2 {
public:
    int climbStairs(int n) {
        array<int,45> arr{1,2};
        if(--n < 2)
            return arr[n];
        for(int i = 2;i <= n;i++) {
            arr[i] = arr[i - 2] + arr[i - 1];
        }
        return arr[n];
    }
};

// 2 -> 2
// 3 -> 3

// 4 -> 5
// 1 + 1 + 1 + 1
// 1 + 1 + 2
// 1 + 2 + 1
// 2 + 1 + 1
// 2 + 2

// 5 -> 8
// 1 + 1 + 1 + 1 + 1
// 1 + 1 + 1 + 2
// 1 + 1 + 2 + 1
// 1 + 2 + 1 + 1
// 2 + 1 + 1 + 1
// 1 + 2 + 2
// 2 + 1 + 2
// 2 + 2 + 1

class Solution3 {
public:
    int climbStairs(int n) {
        int buf[2] = {1,2},res = 0;
        if(--n < 2)
            return buf[n];
        --n;
        while(n--) {
            res = buf[0] + buf[1];
            buf[0] = buf[1];
            buf[1] = res;
        }
        return res;
    }
};

class Solution {
private:
    unordered_map<int,int> map;
public:
    // dfs with memory
    int climbStairs(int n) {
        if(n == 0)
            return 1;
        if(n < 0)
            return 0;
        int res;
        if(map.find(n) != map.end())
            res = map[n];
        else {
            res = climbStairs(n - 1) + climbStairs(n - 2);
            map[n] = res;
        }
        return res;
    }
};

int main() {
    for(int i = 1;i <= 5;i++) {
        cout << Solution().climbStairs(i) << ',';
    }
    cout << endl << Solution().climbStairs(44) << endl;
    return 0;
}