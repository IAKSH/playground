// https://leetcode.cn/problems/sort-colors/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class Solution {
public:
    void sortColors(vector<int>& nums) {
        // shell sort
        for(int gap = nums.size() / 2;gap > 0;gap /= 2) {
            for (int i = gap; i < nums.size(); i++) {
                int temp = nums[i];
                int j;
                for (j = i; j >= gap && nums[j - gap] > temp; j -= gap)
                    nums[j] = nums[j - gap];
                nums[j] = temp;
            }
        }
    }
};

int main() {
    vector<int> nums{2,0,2,1,1,0};
    Solution().sortColors(nums);
    for(const auto& i : nums)
        cout << i << ',';
    cout << "\b \n";
    return 0;
}