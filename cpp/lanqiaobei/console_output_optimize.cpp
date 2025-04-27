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

int main() noexcept {
    double putchar_time = countTime<test_putchar>();
    double cout_time = countTime<test_cout>();
    double cout_opt_time = countTime<test_cout_with_opt>();
    double cout_full_opt_time = countTime<test_cout_with_full_opt>();
    
    system("cls");
    cout << putchar_time << "ms\n";         // 最慢
    cout << cout_time << "ms\n";
    cout << cout_opt_time << "ms\n";        // 最快
    cout << cout_full_opt_time << "ms\n";   // 和cout_opt_time只有误差级区别
    return 0;
}