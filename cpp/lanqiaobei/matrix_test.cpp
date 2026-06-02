#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(0);

    vector<vector<int>> m{{114,514},{1919,810}};
    for(const auto& i : m) {
        for(const auto& j : i)
            cout << j << ' ';
        cout << '\n';
    }
}