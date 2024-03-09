#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>

using namespace std;

int main(){
    int arr1[12]{1,1,2,2,3,3,4,4,5,5,6,6};

    reverse(begin(arr1),end(arr1));

    copy(begin(arr1),end(arr1), ostream_iterator<int>(cout, " "));
    cout << '\n';

    return 0;
}
