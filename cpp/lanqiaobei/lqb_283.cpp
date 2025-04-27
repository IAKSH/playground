// https://www.lanqiao.cn/problems/283/learning/?page=2&first_category_id=1&second_category_id=3&tags=%E5%9B%BD%E8%B5%9B&sort=pass_rate&asc=0

#include <bits/stdc++.h>

using namespace std;

int main() {
    ios::sync_with_stdio(false);

    int n,i,j,len,buf = 0;
    int x = 0,y = 0,dir = 0;// dir 0 1 2 3 => up down left right 
    cin >> n;
    string s;
    bool reading_num = false;
    for(i = 0;i < n;i++) {
        cin >> s;
        len = s.size();
        for(j = 0;j < len;j++) {
            if(reading_num) {
                if(s[j] >= '0' && s[j] <= '9') {

                }
                else {
                    reading_num = false;
                    switch(s[j]) {
                    case ''
                    }
                }
            }
            else {

            }
        }
    }
}