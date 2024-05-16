// https://www.luogu.com.cn/problem/P5788
// AC

// 想到了一种新的方案
// 其实只需要一个栈，并且算法可以随着输入进行
// 简单地说，我们只需要判断当前输入的数是否是栈顶的匹配项，是则记录（这个过程需要循环，要一路秒下去，因为可能有多个匹配），否则直接塞入栈中
// 通过这种方式，我们能保证这个栈永远是单调的

#include <bits/stdc++.h>

using namespace std;

struct Data {
    int val;
    int index;
};

int main() {
    ios::sync_with_stdio(false);

    int n;cin >> n;
    //vector<int> res(n);
    // 好吧，既然确定是GCC，那为什么不用GNU C++ 11呢？
    // 甚至会比堆空间更快，虽然这里的读写密度似乎不是很高
    int res[n] = {0};
    stack<Data> s;

    Data d;
    for(int i = 0;i < n;i++) {
        cin >> d.val;
        d.index = i;
        if(s.empty())
            s.emplace(d);
        else {
            while(!s.empty() && d.val > s.top().val) {
                res[s.top().index] = i + 1;// 因为题目的索引是从1开始的，所以+1
                s.pop();
            }
            s.emplace(d);
        }
    }

    // 栈中剩余的项都不能在其后找到比他更大的项，这些项对应的res也没有被更新，保持为初始的0
    for(const auto& i : res)
        cout << i << ' ';
    return 0;
}