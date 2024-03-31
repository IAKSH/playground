#include <bits/stdc++.h>

using namespace std;

bool isPrime1(int n) noexcept {
    for(int i = 2;i < n;i++) {
        if(n % i == 0)
            return false; 
    }
    return true;
}

bool isPrime2(int n) noexcept {
    int sqrt_val = sqrt(n);
    for(int i = 2;i < sqrt_val;i++) {
        if(n % i == 0)
            return false;
    }
    return true;
}

// 埃氏筛法
bool isPrime3(int n) noexcept {
    if (n <= 1) return false;
    vector<bool> isPrime(n + 1, true);
    isPrime[0] = isPrime[1] = false;
    for (int i = 2; i * i <= n; ++i) {
        if (isPrime[i]) {
            for (int j = i * i; j <= n; j += i) {
                isPrime[j] = false;
            }
        }
    }
    return isPrime[n];
}

// 欧拉筛法
bool isPrime4(int n) noexcept {
    if (n <= 1) return false;
    vector<int> primes;
    vector<bool> isPrime(n + 1, true);
    isPrime[0] = isPrime[1] = false;
    for (int i = 2; i <= n; ++i) {
        if (isPrime[i]) {
            primes.push_back(i);
        }
        for (int j = 0; j < primes.size() && i * primes[j] <= n; ++j) {
            isPrime[i * primes[j]] = false;
            if (i % primes[j] == 0) break;
        }
    }
    return isPrime[n];
}

bool test(function<bool(int)> isPrime) noexcept {
    array<int,25> primes{2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73, 79,83,89,97};
    for(const auto& i : primes) {
        if(!isPrime(i))
            return false;
    }
    return true;
}

int main() noexcept {
    cout << (test(isPrime1) ? "[PASSED]\n" : "[FAILED]\n");
    cout << (test(isPrime2) ? "[PASSED]\n" : "[FAILED]\n");
    cout << (test(isPrime3) ? "[PASSED]\n" : "[FAILED]\n");
    cout << (test(isPrime4) ? "[PASSED]\n" : "[FAILED]\n");
    return 0;
}