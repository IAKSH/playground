#include <iostream>
#include <algorithm>

using namespace std;

/**
 * std::lexicographical_compare
 * 用于比较两个范围的字典序。
*/

/**
 * 下面几个都是和字典序有关的，懒得写示例了
 * next_permutation
 * prev_permutation
 * is_permutation
*/

int main () {
    char one [] = "apple";
    char two [] = "Applement";

    if (lexicographical_compare (one, one + 5, two, two + 5))
        cout << "apple is lexicographically less than Applement";
    else
        cout << "apple is not lexicographically less than Applement";

    return 0;
}
