// 参考：https://oi-wiki.org/contest/io/#%E8%AF%BB%E5%85%A5%E4%BC%98%E5%8C%96
// 我的结论和上述文章不符，至少在我当前的mingw-gcc上
// 关闭ios::sync_with_stdio确实有效果，但是使用putchar逐位输出完全是负优化
// 另外，cin.tie(nullptr)似乎也没有什么效果

#include <bits/stdc++.h>

using namespace std;

template <void(*FUNC)()>
double countTime() noexcept {
    auto start = std::chrono::high_resolution_clock::now();
    FUNC();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
}

/*
void write(int x) {
    if(x < 0) {
        x = -x;
        putchar('-');
    }
    if(x > 9)
        write(x / 10);
    putchar(x % 10 + '0');
}
*/

void write(int x) noexcept {
  static int sta[35];
  int top = 0;
  do {
    sta[top++] = x % 10, x /= 10;
  } while (x);
  while (top) putchar(sta[--top] + 48);  // 48 是 '0'
}

void test_putchar() noexcept {
    for(int i = 0;i < 10000;i++) {
        write(i);
        putchar('\n');
    }
}

void test_cout_with_opt() noexcept {
    ios::sync_with_stdio(false);
    for(int i = 0;i < 10000;i++) {
        cout << i << '\n';
    }
}

void test_cout_with_full_opt() noexcept {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    for(int i = 0;i < 10000;i++) {
        cout << i << '\n';
    }
    cout << flush;
}

void test_cout() noexcept {
    ios::sync_with_stdio(true);
    for(int i = 0;i < 10000;i++) {
        cout << i << '\n';
    }
}

void reset_cout() {
    // 使用一个临时的 ostringstream（默认状态）重置 cout 的格式
    std::cout.copyfmt(std::ostringstream());
    // 可选：清除可能存在的错误状态（如 failbit、badbit）
    std::cout.clear();
}

int main() noexcept {
    double t0,t1,t2,t3;
    t0 = t1 = t2 = t3 = 0;

    constexpr int N = 10;

    for(int i = 0;i < N;i++) {
        reset_cout();
        t0 += countTime<test_putchar>();
        reset_cout();
        t1 += countTime<test_cout>();
        reset_cout();
        t2 += countTime<test_cout_with_opt>();
        reset_cout();
        t3 += countTime<test_cout_with_full_opt>();
    }

    t0 /= N; 
    t1 /= N;
    t2 /= N;
    t3 /= N;
    
    //system("cls");
    system("clear");
    cout << "putchar:\t" << t0 << "ms\n";         // 最慢
    cout << "cout:   \t" << t1 << "ms\n";
    cout << "cout_opt:\t" << t2 << "ms\n";        // 最快
    cout << "cout_full_opt:\t" << t3 << "ms\n";   // 和cout_opt_time只有误差级区别
    return 0;
}