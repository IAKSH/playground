#include <iostream>
#include <chrono>
#include <functional>
#include <random>
#include <array>
#include <algorithm>

template <void(*FUNC)()>
double countTime() noexcept {
    auto start = std::chrono::high_resolution_clock::now();
    FUNC();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
}

int myMax(const int* arr,int i,int j) noexcept {
    int maxn = INT_MIN;
    for(;i < j;i++) {
        if(arr[i] > maxn)
            maxn = arr[i];
    }
    return maxn;
}

std::array<int, 100000> arr;

int myMaxResult = 1;
int stdMaxWithIteratorResult = 2;
int stdMaxWithIndexResult = 3;
int stdSortResult = 4;

void testMyMax() noexcept {
    myMaxResult = myMax(arr.data(),0,arr.size());
}

void testStdMaxWithIterator() noexcept {
    stdMaxWithIteratorResult = *std::max_element(arr.begin(),arr.end());
}

void testStdMaxWithIndex() noexcept {
    stdMaxWithIndexResult = *std::max_element(arr.data(),arr.data() + arr.size());
}

// 注：修改了arr内容
void testStdSort() noexcept {
    std::sort(arr.begin(),arr.end(),std::greater<int>());
    stdSortResult = arr[0];
}

int main() {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(INT_MIN, INT_MAX);
    for (int& num : arr) {
        num = dis(gen);
    }

    std::cout << "my max: \t" <<  countTime<testMyMax>() << "ns\n";
    std::cout << "std::max ite: \t" << countTime<testStdMaxWithIterator>() << "ns\n";
    std::cout << "std::max ind: \t" << countTime<testStdMaxWithIndex>() << "ns\n";
    std::cout << "std::sort: \t" << countTime<testStdSort>() << "ns\n";

    std::cout << "\nresults:\n";
    std::cout << myMaxResult << '\t'
        << stdMaxWithIteratorResult << '\t'
        << stdMaxWithIndexResult << '\t'
        << stdSortResult << '\t';

    return 0;
}

// 结论：max_element无论如何都比手写的O(n)慢
// 但是很明显max_element更通用，成也迭代器败也迭代器
// 大概只是在这种线性容器上输掉了吧，如果是非线性容器，手写的O(n)不也得引入类似迭代器的东西吗
// 但是就算这样，线性容器的性能也非常重要，做题的时候还有时间的话就试试把线性容器的max_element之类的换手写吧
// ~~或者有没有一种可能，其实直接std::sort更快（？~~ 并不，std::sort甚至是O(nlogn)了 (通常情况下