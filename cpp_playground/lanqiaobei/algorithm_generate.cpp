#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>

using namespace std;

class Fibonacci{
    int f1;
    int f2;
public:
    Fibonacci(int start1, int start2){
        f1 = start1;
        f2 = start2;
    }
    int operator()(){
        int r = f1 + f2;
        f1 = f2;
        f2 = r;
        return r;
    }
};

int main(){
    vector<int> v1(10);
    generate(v1.begin(), v1.end(), Fibonacci(0, 1));
    cout<< "0, 1开始前10个斐波那契数列为: " << endl;
    copy(v1.begin(), v1.end(), ostream_iterator<int>(cout, " "));
    
    cout << '\n';

    vector<int> v2(10);
    generate(v1.begin(), v1.end(), [](){
        static int i = 0;
        return i++;
    });
    copy(v1.begin(), v1.end(), ostream_iterator<int>(cout, " "));
    return 0;
}
