/**
 * algorithm库中有一套完整的构建和操作最大堆的函数
 * 应该是为了实现堆排序(std::sort_heap)所设计的
 * std::push_heap (用于在添加一个元素后手动维护堆，通常比make_heap高效，见下面代码)
 * std::pop_heap  (用于在删除一个元素后手动维护堆，通常比make_heap高效)
 * std::make_heap 不一定创建的是大顶堆，有第三个参数，为greater<>()则生成小顶堆，上二同
 * std::sort_heap
 * std::is_heap (C++ 11)
 * std::is_heap_until (C++ 11)
 * 具体怎么用，懒得写了
*/

#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

void print(vector<int>& vc) {
    for (auto i : vc) {
        cout << i << " ";
    }
    cout << endl;
}

int main() {
    vector<int> vc{20, 30, 40, 10};
    make_heap(vc.begin(), vc.end());
    cout << "Initial Heap: ";
    print(vc);

    vc.push_back(50);
    cout << "Heap just after push_back(): ";
    print(vc);

    push_heap(vc.begin(), vc.end());
    cout << "Heap after push_heap(): ";
    print(vc);

    return 0;
}
