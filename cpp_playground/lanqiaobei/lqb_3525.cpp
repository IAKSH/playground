/**
 * https://www.lanqiao.cn/problems/3525/learning/?page=3&first_category_id=1&tags=2023
 * 6超时
*/


#include <bits/stdc++.h>

using namespace std;

int main() noexcept {
    int n;cin >> n;
    vector<int> v(n);
    for(auto& i : v)
        cin >> i;
    
    vector<pair<int,int>> results;
    for(int i = 0;i < n;i++) {
        for(int j = i + 1;j < n;j++) {
            // 超时的原因似乎很大程度在这里，__gcd还是太慢了
            // 应该可以做质数筛，来分解质因数
            if(__gcd(v[i],v[j]) > 1)
                results.emplace_back(pair<int,int>(i + 1,j + 1));
        }
    }

    // 这部分其实应该是可以通过两次O(n)遍历得到最终结果的
    int min_i = min_element(results.begin(),results.end(),[](const pair<int,int>& p1,const pair<int,int>& p2){
        return p1.first < p2.first;
    })->first;
    remove_if(results.begin(),results.end(),[&](const pair<int,int>& p){
        return p.first != min_i;
    });

    if(results.size() != 1) {
        int min_j = min_element(results.begin(),results.end(),[](const pair<int,int>& p1,const pair<int,int>& p2){
            return p1.second < p2.second;
        })->first;
        remove_if(results.begin(),results.end(),[&](const pair<int,int>& p){
            return p.second != min_j;
        });
    }

    const auto& result = results[0];
    cout << result.first << ' ' << result.second << '\n';
    return 0;
}