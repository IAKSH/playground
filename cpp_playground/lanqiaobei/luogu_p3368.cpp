// https://www.luogu.com.cn/problem/P3368
// 树状数组 (二元索引树)
// 7AC 3TLE

#include <bits/stdc++.h>

using namespace std;

class BITree {
public:
    BITree(vector<int>& nums) : nums(nums) {
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

    void update(int index,int v) {
		add(index + 1,v - nums[index]);
		nums[index] = v;
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

    int get(int index){
	    //int res = 0;
	    //for(int i = index;i >= 1;i -= lowbit(i))
        //    res += tree[i];
	    //return res;

        return prefix_sum(index) - prefix_sum(index - 1);
    }

private:
    int lowbit(int x) {
		return x & (-x);
	}

    vector<int>& nums;
    vector<int> tree;
};

int main() {
    ios::sync_with_stdio(false);
    
    int n,m,a,b,c,d,i,j;
    cin >> n >> m;

    vector<int> v(n);
    for(auto& val : v)
        cin >> val;

    BITree bitree(v);
    
    for(i = 0;i < m;i++) {
        cin >> a;
        if(a == 1) {
            cin >> b >> c >> d;
            for(j = b;j <= c;j++)
                bitree.add(j,d);
        }
        else {
            cin >> b;
            cout << bitree.get(b) << '\n';
        }
    }

    return 0;
}