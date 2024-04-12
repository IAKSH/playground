// https://www.lanqiao.cn/problems/104/learning/

#include <bits/stdc++.h>

using namespace std;

/*
      祥 瑞 生 辉
  +   三 羊 献 瑞
-------------------
   三 羊 生 瑞 气

   A B C D
+  E F G H
------------
 E F C H X
*/

bool nequal(std::initializer_list<int> list) noexcept {
    for(int i = 0;i < list.size();i++) {
        for(int j = 0;j < list.size();j++) {
            if(i != j && *(list.begin() + i) == *(list.begin() + j))
                return false;
        }
    }
    return true;
}

int main() noexcept {
    for(int a = 0;a <= 9;a++) {
        for(int b = 0;b <= 9;b++) {
            for(int c = 0;c <= 9;c++) {
                for(int d = 0;d <= 9;d++) {
                    for(int e = 0;e <= 9;e++) {
                        for(int f = 0;f <= 9;f++) {
                            for(int g = 0;g <= 9;g++) {
                                for(int h = 0;h <= 9;h++) {
                                    for(int x = 0;x <= 9;x++) {
                                        if((a*1000 + b*100 + c*10 + d + e*1000 + f*100 + g*10 + h == e*10000 + f*1000 + c*100 + h*10 + x)
                                            && nequal({a,b,c,d,e,f,g,h,x})) {
                                            cout << a << b << c << d << endl << e << f << g << h << endl << e << f << c << h << x << endl;
                                            cout << "EFGH: " << e << f << g << h << endl << endl;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}