#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;


int main() noexcept {
    string s1 = "nihaonihaoma?";
    string s2 = "ih";

    string::iterator it1 = find_first_of(s1.begin(),s1.end(),s2.begin(),s2.end());
    cout << &*it1 << '\n';
    string::iterator it2 = find_end(s1.begin(),s1.end(),s2.begin(),s2.end());
    cout << &*it2 << '\n';
    
    char str[it2 - it1];
    copy(it1,it2,str);

    for(const auto& c : str) {
        cout << c;
    }
    cout << endl;
    return 0;
}