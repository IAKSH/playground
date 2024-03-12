//https://www.dotcpp.com/oj/problem3151.html

#include <algorithm>
#include <vector>
#include <cstdio>
#include <cmath>

/*
2 3 0 100 10 10 10 10 0 2 20 3 0 10 20 10 10 20 20 10 20
YES
NO
*/

// n个飞机有许多种降落先后顺序组合，所有可能的组合构成了一个树
// 在一个树中做dfs

// 有点反直觉的是，降落过程中不计算油耗，或者说降落过程中燃油耗尽并不影响降落

/**
 * https://www.dotcpp.com/oj/submit_status.php?sid=15728653
 * 运行时间: 6ms    消耗内存: 1272KB
*/

struct Plane{
    int t,d,l;
    Plane(int t,int d,int l)
        : t(t),d(d),l(l)
    {}
};

bool dfs(const std::vector<Plane>& planes,std::vector<short>& flags,int dt) noexcept {
    //if(std::equal(std::begin(flags),std::end(flags),true)) {
    //    return true;
    //}
    bool all_checked{ true };
    for(const auto& val : flags) {
        if(!val) {
            all_checked = false;
            break;
        }
    }
    if(all_checked) {
        return true;
    }

    for(int i = 0;i < planes.size();i++) {
        const Plane& pi = planes[i];
        if(!flags[i] && dt <= pi.t + pi.d) {
            flags[i] = true;
            if(dfs(planes,flags,std::max(pi.t,dt) + pi.l)) {
                return true;
            }
            flags[i] = false; 
        }
    }
    return false;
}

int main() noexcept {
    int t;
    scanf("%d",&t);

    while(t--) {
        std::vector<Plane> planes;
        int n;
        scanf("%d",&n);
        for(int i = 0;i < n;i++) {
            int t,d,l;
            scanf("%d%d%d",&t,&d,&l);
            planes.emplace_back(t,d,l);
        }
        std::vector<short> plane_checked_flags(n,0);
        printf(dfs(planes,plane_checked_flags,0) ? "YES\n" : "NO\n");
    }
    return 0;
}