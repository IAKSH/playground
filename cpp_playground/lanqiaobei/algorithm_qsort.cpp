#include <iostream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <random>
#include <iterator>

using namespace std;

int comp(const void* m,const void* n) noexcept {
    return *(const int*)m - *(const int*)n;
}

int main() noexcept {
    vector<int> v{1,2,3,4,5,6,7,8,9,0};
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(v.begin(), v.end(), default_random_engine(seed));

    std::copy(v.begin(),v.end(), std::ostream_iterator<int>(cout, " "));
    cout << '\n';
    std::qsort(v.data(),v.size(),sizeof(int),comp);
    std::copy(v.begin(),v.end(), std::ostream_iterator<int>(cout, " "));
}