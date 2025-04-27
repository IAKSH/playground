// https://leetcode.cn/problems/partition-labels/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> partitionLabels(string s) {
        int last_index[26];
        for(int i = 0;i < s.size();i++) {
            last_index[s[i] - 'a'] = i;
        }
        
        vector<int> results;
        int l = 0;
        int r = 0;
        for(int i = 0;i <= s.size();i++) {
            if(i > r) {
                results.emplace_back(r + 1 - l);
                l = r + 1;
            }
            if(i < s.size())
                r = max(r,last_index[s[i] - 'a']);
        }

        return results;
    }
};

int main() {
    cout << '{';
    for(const auto& i : Solution().partitionLabels("ababcbacadefegdehijhklij"))
        cout << i << ',';
    cout << "\b}\n";
    return 0;
}