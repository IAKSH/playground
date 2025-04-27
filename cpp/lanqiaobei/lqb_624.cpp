// https://www.lanqiao.cn/problems/624/learning/?page=1&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

const float A1 = 2.3f;
const float A2 = 2.5f;
const float B1 = 6.4f;
const float B2 = 3.1f;
const float C1 = 5.1f;
const float C2 = 7.2f;

int main() {
    // 海伦公式
    float a = sqrt(pow(A1 - B1,2) + pow(A2 - B2,2));
    float b = sqrt(pow(A1 - C1,2) + pow(A2 - C2,2));
    float c = sqrt(pow(B1 - C1,2) + pow(B2 - C2,2));
    float p = (a + b + c) / 2.0f;
    printf("%.3f\n",sqrt(p * (p - a) * (p - b) * (p - c)));
    return 0;
}