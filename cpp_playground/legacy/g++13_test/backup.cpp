#include <iostream>
#include <iterator>
#include <vector>
#include <ranges>

using namespace std;

int removeElement(vector<int>& nums, int val) noexcept {
    for(auto ite = begin(nums);ite != end(nums);ite++) {
        if(*ite == val)
            nums.erase(ite);
    }
    return nums.size();
}

int main() noexcept {
    std::vector<int> vec {1,4,5,2,3,5,6,7,8,1,2,3,0};
    removeElement(vec,3);

    for(auto& item : vec) {
        std::cout << item << '\t';
    }

    std::cout << std::endl;
    return 0;
}