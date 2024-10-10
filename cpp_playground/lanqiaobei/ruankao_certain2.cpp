#include <bits/stdc++.h>

using namespace std;

array<array<int,4>,4> arr {
    array<int,4>{14,11,13,17},
    array<int,4>{9,7,2,9},
    array<int,4>{4,7,10,15},
    array<int,4>{15,10,5,13}
};

vector<int> min_path;
int minn = INT_MAX;

void dfs(array<bool,4> mask,vector<int> path,int x,int depth) {
    if(depth == 3) {
        if(x < minn) {
            minn = x;
            min_path = path;
        }
    }
    else {
        for(int i = 0;i < 4;i++) {
            if(!mask[i]) {
                auto new_mask = mask;
                new_mask[i] = 1;
                vector<int> new_path = path;
                new_path.emplace_back(i);
                dfs(new_mask,new_path,x + arr[depth + 1][i],depth + 1);
            }
        }
    }
}

int main() {
    for(int i = 0;i < 4;i++) {
        array<bool,4> mask;
        fill(mask.begin(),mask.end(),false);
        mask[i] = 1;
        dfs(mask,vector<int>{i},arr[0][i],0);
    }
    cout << "minn = " << minn << '\n';
    for(const auto& i : min_path)
        cout << i << ',';
    cout << "\b \n";
    return 0;
}

