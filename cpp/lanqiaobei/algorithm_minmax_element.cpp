#include<iostream>
#include<algorithm>
#include<vector>

using namespace std;

int main() {
    vector<int> vi = {5, 3, 4, 4, 3, 5, 3};
    pair<vector<int>::iterator, vector<int>::iterator> mnmx;

    mnmx = minmax_element(vi.begin(), vi.begin() + 4);

    cout << "The minimum value position obtained is: ";
    cout << mnmx.first - vi.begin() << endl;
    cout << "The maximum value position obtained is: ";
    cout << mnmx.second - vi.begin() << endl;

    cout << endl;

    mnmx = minmax_element(vi.begin(), vi.end());

    cout << "The minimum value position obtained is: ";
    cout << mnmx.first - vi.begin() << endl;
    cout << "The maximum value position obtained is: ";
    cout << mnmx.second - vi.begin() << endl;

    return 0;
}
