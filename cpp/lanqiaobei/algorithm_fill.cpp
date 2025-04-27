#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main() noexcept {
    vector<int> v(3);
    cout << "size = " << v.size() << '\n';
    fill(v.begin(),v.end(),114);
    cout << "size = " << v.size() << '\n';
    for(const auto& i : v) {
        cout << i << '\n';
    }
    return 0;
}