// https://leetcode.cn/problems/find-all-anagrams-in-a-string/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> v;
        if(p.size() > s.size())
            return v;
        array<int,26> arr{0};
        for(int i = 0;i < p.size();i++)
            ++arr[p[i] - 'a'];
        for(int i = 0;i <= s.size() - p.size();i++) {
            if(check(s.begin() + i,p,arr))
                v.emplace_back(i);
        }
        return v;
    }

private:
    bool check(string::iterator begin,const string& p,const array<int,26>& arr) {
        array<int,26> cur_arr{0};
        for(int i = 0;i < p.size();i++) {
            ++cur_arr[*(begin + i) - 'a'];
        }
        return cur_arr == arr;
    }
};

int main() {
    for(const auto& i : Solution().findAnagrams("abab","ab"))
        cout << i << ',';
    cout << "\b \n";
    return 0;
}