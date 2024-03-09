#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;


int main() noexcept {
    int arr1[13] = {0,1,2,3,4,5,6,3,4,-1,-2,-3,-4};
    
    for_each(begin(arr1),end(arr1),[](int& i){
        i *= (i % 2 == 0); 
    });

    for_each(begin(arr1),end(arr1),[](int i){
        cout << i << ' ';
    });
    cout << '\n';

    return 0;
}