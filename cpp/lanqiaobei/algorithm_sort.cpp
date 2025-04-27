#include <iostream>
#include <algorithm>
#include <vector>
#include <iterator>
#include <chrono>
#include <random>

using namespace std;

template <typename T>
void println(T& t) {
    copy(begin(t),end(t), ostream_iterator<int>(cout, " "));
    cout << '\n';
}

template <typename T>
void __shuffle(T& t) {
    static unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(begin(t), end(t), default_random_engine(seed));
}

int main() noexcept {
    vector<int> v{1,434,5,3425,-5,235,34,34,2345,534,34,2,1,6757,23};

    __shuffle(v);
    cout << "shuffled:\t";
    println(v);
    cout << "intro sort:\t";
    sort(begin(v),end(v));
    println(v);
    cout << '\n';
    
    __shuffle(v);
    cout << "shuffled:\t";
    println(v);
    cout << "stable sort:\t";
    stable_sort(begin(v),end(v));
    println(v);
    cout << '\n';
    
    __shuffle(v);
    cout << "shuffled:\t";
    println(v);
    cout << "heap sort:\t";
    make_heap(begin(v),end(v));
    sort_heap(begin(v),end(v));
    println(v);
    cout << '\n';
    
    //is_sorted
    __shuffle(v);
    cout << "shuffled:\t";
    println(v);
    cout << "sorted? " << (is_sorted(begin(v),end(v),[](int a,int b){return a < b;}) ? "true" : "false") << '\n';
    cout << "heap sort:\t";
    make_heap(begin(v),end(v));
    sort_heap(begin(v),end(v));
    println(v);
    cout << "sorted? " << (is_sorted(begin(v),end(v)) ? "true" : "false") << '\n';

    // 略：
    // is_sorted_until
    // partial_sort
    // partial_sort_copy

    return 0;
}