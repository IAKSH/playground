/**
 * https://www.lanqiao.cn/problems/1457/learning/?page=2&first_category_id=1&second_category_id=3&tags=2021
 * 只能过三例
 * 2错误，5超时
*/

#include <bits/stdc++.h>

using namespace std;

int count(int layer,int i) noexcept {
    int res = 0;
    for(layer--;layer >= 0;layer--) {
        res += layer + 1;
    }
    return res + i + 1;
}

int main() noexcept {
    int n,cnt = 0;
    cin >> n;

    deque<int> last_line{1};
    deque<int> line;
    int layer,i;
    
    for(layer = 1;true;layer++) {
        // 每层半边个数 = floor(layer / 2) + 1
        line.clear();
        line.resize(layer / 2 + 1);
        for(i = 0;i < layer / 2 + 1;i++) {
            if(layer % 2 == 0)
                line[i] = ((i < 1) ? 0 : last_line[i - 1]) + ((i >= (layer - 1) / 2 + 1) ? last_line[i - 1] : last_line[i]);
            else
                line[i] = ((i < 1) ? 0 : last_line[i - 1]) + last_line[i];
            if(line[i] == n) {
                cout << count(layer,i) << '\n';
                return 0;
            }
        }
        last_line = line;
    }

    return 1;
}