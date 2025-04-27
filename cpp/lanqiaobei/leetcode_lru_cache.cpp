// https://leetcode.cn/problems/lru-cache/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class LRUCache {
public:
    LRUCache(int capacity) : max_size(capacity) {}

    int get(int key) {
        if (caches_map.count(key) > 0) {
            auto ptr = caches_map[key];
            // 移动ptr到最前
            move_to_front(ptr);
            return ptr->value;
        } else {
            return -1;
        }
    }

    void put(int key, int value) {
        if (caches_map.count(key) > 0) {
            auto ptr = caches_map[key];
            ptr->value = value;
            // 移动ptr到最前
            move_to_front(ptr);
        } else if (used_size < max_size) {
            Node* new_node = nullptr;
            if (!head) {
                new_node = new Node(key, value, nullptr, nullptr);
                head = tail = new_node;
            } else {
                new_node = new Node(key, value, nullptr, head);
                head->front = new_node;
                head = new_node;
            }
            ++used_size;
            caches_map[key] = new_node;
        } else {
            // 最后一个节点修改key和value，然后移动到最前，同时更新caches_map中的映射
            Node* last_node = tail;
            caches_map.erase(last_node->key);
            last_node->key = key;
            last_node->value = value;
            caches_map[key] = last_node;
            move_to_front(last_node);
        }
    }

private:
    struct Node {
        int key, value;
        Node *front, *next;
        Node(int key, int value, Node* front, Node* next) : key(key), value(value), front(front), next(next) {}
    };

    void move_to_front(Node* node) {
        if (node == head) return;
        // 断开node
        if (node->front) node->front->next = node->next;
        if (node->next) node->next->front = node->front;
        if (node == tail) tail = node->front;
        // 将node移动到最前
        node->next = head;
        node->front = nullptr;
        if (head) head->front = node;
        head = node;
        if (!tail) tail = node;
    }

    Node *head = nullptr, *tail = nullptr;
    unordered_map<int, Node*> caches_map;
    const int max_size;
    int used_size = 0;
};

bool test1() {
    vector<int> res;
    LRUCache lru(2);
    lru.put(1,1);
    lru.put(2,2);
    res.emplace_back(lru.get(1));
    lru.put(3,3);
    res.emplace_back( lru.get(2));
    lru.put(4,4);
    res.emplace_back(lru.get(1));
    res.emplace_back(lru.get(3));
    res.emplace_back(lru.get(4));
    return res == vector<int>{1,-1,-1,3,4};
}

bool test2() {
    vector<int> res;
    LRUCache lru(2);
    lru.put(2,1);
    lru.put(2,2);
    res.emplace_back(lru.get(2));
    lru.put(1,1);
    lru.put(4,1);
    res.emplace_back(lru.get(2));
    return res == vector<int>{2,-1};
}

bool test3() {
    vector<int> res;
    LRUCache lru(2);
    lru.put(2,1);
    lru.put(1,1);
    lru.put(2,3);
    lru.put(4,1);
    res.emplace_back(lru.get(1));
    res.emplace_back(lru.get(2));
    return res == vector<int>{-1,3};
}

int main() {
    for(auto func : {test1,test2,test3})
        cout << (func() ? "AC" : "WA") << '\n';
    return 0;
}