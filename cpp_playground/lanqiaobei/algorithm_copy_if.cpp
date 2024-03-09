#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;


int main() noexcept {
    int arr1[13] = {0,1,2,3,4,5,6,3,4,-1,-2,-3,-4};
    int arr2[13] = {0};

    copy_if(begin(arr1),end(arr1),begin(arr2),[](int i){return i % 2 == 0;});
    for(const auto& i : arr2) {
        cout << i << '\n';
    }

    return 0;
}