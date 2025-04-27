#include <iostream>
#include <iterator>
#include <vector>
#include <algorithm>

using namespace std;

int main(){
    int arr1[12]{1,1,2,2,3,3,4,4,5,5,6,6};
    int arr2[12]{0};

    reverse_copy(begin(arr1),end(arr1),begin(arr2));

    copy(begin(arr1),end(arr1), ostream_iterator<int>(cout, " "));
    cout << '\n';
    copy(begin(arr2),end(arr2), ostream_iterator<int>(cout, " "));
    cout << '\n';

    return 0;
}
