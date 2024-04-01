// https://www.lanqiao.cn/problems/182/learning/?page=2&first_category_id=1&second_category_id=3&difficulty=20


#ifdef OLD_VER
/**
 * 这部分代码是很明显有缺陷的
 * 只是能过目前的这几个示例罢了
*/
#include <bits/stdc++.h>

using namespace std;

int getCircleLength(const vector<int>& v,int i) noexcept {
    int cnt = 1;
    int mv_i = v[v[i] - 1];
    while(mv_i != v[i]) {
        mv_i = v[mv_i - 1];
        ++cnt;
    }
    return cnt;
}

int main() noexcept {
    int n;cin >> n;
    vector<int> v(n);
    for(auto& i : v)
        cin >> i;

    int max_circle_length = INT_MIN;
    for(int i = 0;i < n;i++) {
        int circle_len = 0;
        max_circle_length = max(max_circle_length,getCircleLength(v,i));
    }

    cout << max_circle_length << '\n';
    return 0;
}
#else
/**
 * 虽然也挺慢的，无脑记忆化
 * 懒得再改了
*/
#include <bits/stdc++.h>

using namespace std;

static int n;
static vector<int> v;
static vector<bool> memory,tmp_memory;

int getCircleLength(int i) noexcept {
    int cnt = 1;
    int mv_i = v[v[i] - 1];
    fill(tmp_memory.begin(),tmp_memory.end(),false);
    while(mv_i != v[i]) {
        if(tmp_memory[mv_i] || memory[mv_i])
            return 0;
        tmp_memory[mv_i] = true;
        mv_i = v[mv_i - 1];
        ++cnt;
    }
    for(int j = 0;j < 0;j++)
        if(tmp_memory[j])
            memory[j] = true;
    return cnt;
}

int main() noexcept {
    cin >> n;
    v.resize(n);
    memory.resize(n);
    tmp_memory.resize(n);
    for(auto& i : v)
        cin >> i;

    int max_circle_length = INT_MIN;
    for(int i = 0;i < n;i++) {
        int circle_len = 0;
        max_circle_length = max(max_circle_length,getCircleLength(i));
    }

    cout << max_circle_length << '\n';
    return 0;
}
#endif