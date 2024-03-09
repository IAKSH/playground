#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main() noexcept {
    vector<int> v1(9);
    vector<int> v2{1,2,3,4,5,6,7,8,9};
    copy(v2.begin(),v2.end(),v1.begin());
    cout << "v1.size() = " << v1.size() << '\n';
    for(const auto& i : v1) {
        cout << i << '\n';
    }
    return 0;
}