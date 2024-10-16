// https://leetcode.cn/problems/longest-substring-without-repeating-characters/submissions/573300495/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int maxn = 0;
        deque<char> dq;
        for(const auto& c : s) {
            while(!dq.empty() && find(dq.begin(),dq.end(),c) != dq.end())
                dq.pop_front();
            dq.emplace_back(c);
            maxn = max(maxn,static_cast<int>(dq.size()));
        }
        return maxn;
    }
};

int main() {
    cout << Solution().lengthOfLongestSubstring("bbbb") << '\n';
    return 0;
}