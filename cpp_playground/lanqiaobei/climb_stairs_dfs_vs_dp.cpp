#include <iostream>
#include <chrono>

class DFSSolution {
private:
    int n;
    int res = 0;
    void dfs(int cur,int clamb) noexcept {
        cur += clamb;
        if(cur == n) {
            res++;
            return;
        }
        if(cur > n) {
            return;
        }
        dfs(cur,1);
        dfs(cur,2);
    }

public:
    int climbStairs(int n) noexcept {
        this->n = n;
        dfs(0,1);
        dfs(0,2);
        return res;
    }
};

class DPSolution {
public:
    int climbStairs(int n) {
        int buffer[46] = {1,1};
        for(int i = 2;i < 46;i++) buffer[i] = buffer[i - 1] + buffer[i - 2];
        return buffer[n];
    }
};

int main() noexcept {
    auto t1 = std::chrono::high_resolution_clock::now();
    DFSSolution dfs;
    std::cout << dfs.climbStairs(44) << '\n';
    auto t2 = std::chrono::high_resolution_clock::now();

    auto t3 = std::chrono::high_resolution_clock::now();
    DPSolution dp;
    std::cout << dp.climbStairs(44) << '\n';
    auto t4 = std::chrono::high_resolution_clock::now();

    std::cout << "dfs: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";
    std::cout << "dp: " << std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3).count() << " ms\n";

    return 0;
}