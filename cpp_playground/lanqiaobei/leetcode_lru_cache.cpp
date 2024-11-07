// https://leetcode.cn/problems/lru-cache/description/?envType=study-plan-v2&envId=top-100-liked

#include <bits/stdc++.h>

using namespace std;

class LRUCache {
public:
    LRUCache(int capacity) {
        caches.resize(capacity);
    }
    
    int get(int key) {
        auto it = find_if(caches.begin(),caches.end(),[&](const CacheBlock& cb){
            return cb.key == key;
        });
        if(it != caches.end()) {
            for(auto& cb : caches)
                ++cb.last_update;
            it->last_update = 0;
            return it->val;
        }
        else
            return -1;
    }
    
    void put(int key, int value) {
        auto it = find_if(caches.begin(),caches.end(),[&](const CacheBlock& cb){
            return cb.key == key;
        });
        if(it != caches.end()) {
            for(auto& cb : caches)
                ++cb.last_update;
            *it = CacheBlock(key,value);
        }
        else if(used_size < caches.size()) {
            for(auto& cb : caches)
                ++cb.last_update;
            caches[used_size++] = CacheBlock(key,value);
        }
        else {
            auto min_it = min_element(caches.begin(),caches.end(),[](const CacheBlock& cb1,const CacheBlock& cb2){
                return cb1.last_update > cb2.last_update;
            });
            for(auto& cb : caches)
                ++cb.last_update;
            *min_it = CacheBlock(key,value);
        }
    }

private:
    struct CacheBlock {
        int key,val,last_update;
        CacheBlock() : key(-1),val(-1),last_update(0) {}
        CacheBlock(int key,int val) : key(key),val(val),last_update(0) {}
    };

    vector<CacheBlock> caches;
    int used_size = 0;
};

void test1() {
    LRUCache lru(2);
    lru.put(1,1);
    lru.put(2,2);
    cout << lru.get(1) << '\n';
    lru.put(3,3);
    cout << lru.get(2) << '\n';
    lru.put(4,4);
    cout << lru.get(1) << '\n';
    cout << lru.get(3) << '\n';
    cout << lru.get(4) << '\n';
}

void test2() {
    LRUCache lru(2);
    lru.put(2,1);
    lru.put(2,2);
    cout << lru.get(2) << '\n';
    lru.put(1,1);
    lru.put(4,1);
    cout << lru.get(2) << '\n';
}

void test3() {
    LRUCache lru(2);
    lru.put(2,1);
    lru.put(1,1);
    lru.put(2,3);
    lru.put(4,1);
    cout << lru.get(1) << '\n';
    cout << lru.get(2) << '\n';
}

int main() {
    test3();
    return 0;
}