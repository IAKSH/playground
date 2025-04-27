#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;


int main() noexcept {
    int arr1[13] = {0,1,2,3,4,5,6,3,4,-1,-2,-3,-4};
    
    replace(begin(arr1),end(arr1),1,114);
    for(const auto& i : arr1) {
        cout << i << ' ';
    }
    cout << endl;

    return 0;
}