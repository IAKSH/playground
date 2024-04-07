// https://www.lanqiao.cn/problems/2410/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023

#include <bits/stdc++.h>

using namespace std;

vector<string> v;

int dfs(int x,int y,int len) noexcept {
	//cout << v[y][x] << '\n';
	if(x >= 0 && x < 60 && y >= 0 && y < 30 && v[y][x] == '1') {
		v[y][x] = '0';
        // 向四面搜索
		len = dfs(x + 1,y,len);
        len = dfs(x - 1,y,len);
        len = dfs(x,y + 1,len);
        len = dfs(x,y - 1,len);
        // 这个过程实际上将len + 1的1累加起来了，len就是之前所有节点的+1之和
        return len + 1;
	}
    return len;
}

int main() noexcept {
	ifstream ifs("../lqb_2410.in",ios::in);
	
	while(!ifs.eof()) {
		v.emplace_back("");
		getline(ifs,v.back());
	}
	ifs.close();
    int maxn = INT_MIN;
	for(int i = 0;i < 30;i++) {
		for(int j = 0;j < 60;j++) {
            if(v[i][j] == '1')
			    maxn = max(dfs(j,i,0),maxn);
		}
	}
	cout << maxn << '\n';
	return 0;
}