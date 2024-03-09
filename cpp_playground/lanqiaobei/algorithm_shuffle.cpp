#include <iostream>      // std::cout
#include <algorithm>     // std::shuffle
#include <array>         // std::array
#include <random>        // std::default_random_engine
#include <chrono>        // std::chrono::system_clock

/*
 * 需要注意的是，std::random_shuffle 已在 C++17 中被弃用，并在 C++14 中被标记为过时。
 * 这是因为 std::random_shuffle 通常依赖于 std::rand，而 std::rand 现在也在讨论是否应被弃用。
 * 建议使用 <random> 中的类来替换 std::rand。
 * 因此，建议使用 std::shuffle 而不是 std::random_shuffle。
*/

using namespace std;

int main() {
    array<int,10> foo;
    generate(foo.begin(),foo.end(),[](){
        static int i = 0;
        return i++;
    });

    // obtain a time-based seed:
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();

    shuffle(foo.begin(), foo.end(), default_random_engine(seed));

    cout << "shuffled elements:";
    for (int& x: foo) cout << ' ' << x;
    cout << '\n';

    return 0;
}
