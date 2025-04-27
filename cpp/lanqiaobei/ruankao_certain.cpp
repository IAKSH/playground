#include <bits/stdc++.h>

using namespace std;

array<array<float,5>,3> arr {
    array<float,5>{3.8,4.1,4.8,6.0,6.6},
    array<float,5>{4.0,4.2,5.0,6.0,6.6},
    array<float,5>{4.8,6.4,6.8,7.8,7.8}
};

float maxn = 0;

void dfs(int depth,int i_sum,float sum) {
    if(depth == 2) {
        maxn = max(maxn,sum);
        return;
    }
    else {
        for(int i = 0;i < 4 - i_sum;i++) {
            dfs(depth + 1,i_sum + i,sum + arr[depth + 1][i]);
        }
    }
}

int main() {
    for(int i = 0;i < arr[0].size();i++)
        dfs(0,i,arr[0][i]);
    cout << maxn << '\n';
    return 0;
}