// https://leetcode.cn/problems/min-stack/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class MinStack {
public:
    MinStack() {}
    
    void push(int val) {
        stack.push(val);
        if(val < min_val) {
            min_val = val;
            min_index = stack.size() - 1;
        }
    }
    
    void pop() {
        stack.pop();
        if(min_index == stack.size()) {
            min_val = INT_MAX;
            auto s = stack;
            while(!s.empty()) {
                if(s.top() < min_val) {
                    min_val = s.top();
                    min_index = s.size() - 1;
                }
                s.pop();
            }
        }
    }
    
    int top() {
        return stack.top();
    }
    
    int getMin() {
        return min_val;
    }

private:
    std::stack<int> stack;
    int min_val = INT_MAX;
    int min_index = -1;
};

int main() {
    MinStack minStack;
    minStack.push(-2);
    minStack.push(0);
    minStack.push(-3);
    cout << minStack.getMin() << '\n';
    minStack.pop();
    cout << minStack.top() << '\n';
    cout << minStack.getMin() << '\n';
    return 0;
}