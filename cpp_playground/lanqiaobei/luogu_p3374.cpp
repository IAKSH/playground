// https://www.luogu.com.cn/problem/P3374
// 树状数组 (二元索引树)
// AC

// 一个比较不错的原理解释
// https://www.bilibili.com/video/BV1ce411u7qP

#include <bits/stdc++.h>

using namespace std;

// 纯粹的树状数组
// 适合处理点修改和区间求和

class BITree {
public:
    BITree(vector<int>& nums) {
        int len = nums.size();
        tree.resize(len + 1);
		for(int i = 0; i < len; i++) {
			add(i + 1, nums[i]);
		}
    }

    void add(int index,int val) {
        int len = tree.size();
		while (index < len) {
			tree[index] += val;
			index += lowbit(index);
		}
	}

    int prefix_sum(int index) {
		int sum = 0;
		while (index > 0) {
			sum += tree[index];
			index -= lowbit(index);
		}
		return sum;
	}

    int range_sum(int left,int right) {
		return prefix_sum(right) - prefix_sum(left - 1);
	}

private:
    int lowbit(int x) {
		return x & (-x);
	}

    vector<int> tree;
};

int main() {
    ios::sync_with_stdio(false);
    
    int n,m,i,z,x,y;
    cin >> n >> m;

    vector<int> v(n);
    for(i = 0;i < n;i++)
        cin >> v[i];

    BITree bitree(v);

    for(i = 0;i < m;i++) {
        cin >> z >> x >> y;
        if(z == 1)
            bitree.add(x,y);
        else
            cout << bitree.range_sum(x,y) << '\n';
    }

    return 0;
}