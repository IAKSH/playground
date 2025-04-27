#include <iostream>
#include <queue>

// 优先队列：自动排序
// std::priority_queue是一个容器适配器，并不是一个具体的容器，std::priority_queue需要建立在某种底层容器的基础上（比如std::vector或std::deque)
// std::priority_queue的默认底层容器是std::vector
// std::priority_queue的底层容器必须有back()，push_back()和pop_back()，且必须提供随机访问迭代器。
// 默认用std::less比较，可换std::greater或自定比较函数
// 没有迭代器，不能迭代器遍历
// 可以以O(1)获取最大/最小值
// 还有更多用法，没有看完。
// 这个插入时自动排序也许有用

// std::priority_queue的插入的时间复杂度是O(nlogn)
// 查找极值的时间复杂度是O(1)
// std::priority_queue的排序实现通常是二叉堆

struct Data {
    int x,y;
    Data(int x) : x(x), y(random()) {}
};

std::ostream& operator<<(std::ostream& os,const Data& data) {
    os << "data={x=" << data.x << ",y=" << data.y << "}";
    return os;
}

template <typename T>
void test_p_queue(T& t) noexcept {
    std::cout << "input: ";
    for(int i = 0;i < 10;i++) {
        std::cout << i << ' ';
        t.emplace(i);
    }
    std::cout << '\n';

    std::cout << "output: ";
    while(!t.empty()) {
        std::cout << t.top() << ' ';
        t.pop();
    }
    std::cout << '\n';
}

int main() noexcept {
    std::priority_queue<int> less_p_queue;
    std::priority_queue<int,std::vector<int>,std::greater<int>> greater_p_queue;

    auto comp = [](const Data& m,const Data& n){return m.y > n.y;};
    // lambda在模板参数里需要decltype。并且最后作为参数塞给std::priority_queue的构造（正常函数也要）
    std::priority_queue<Data,std::deque<Data>,decltype(comp)> struct_p_queue(comp);
    
    test_p_queue(less_p_queue);
    test_p_queue(greater_p_queue);
    test_p_queue(struct_p_queue);

    return 0;
}
