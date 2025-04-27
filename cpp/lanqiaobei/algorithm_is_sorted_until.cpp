#include <iostream> // std::cout
#include <algorithm> // std::partition, std::partition_point
#include <vector> // std::vector

/*
 * std::is_sorted_until 用于查找范围 [first, last) 中第一个未排序的元素。
 * 它返回一个迭代器，指向范围中第一个未排序的元素，因此返回的迭代器和 first 之间的所有元素都是已排序的。
 * 它还可用于计算范围内已排序元素的总数。
 */

// 还有一个std::is_sorted，见algorithm_is_sorted.cpp

bool IsOdd (int i) { return (i%2)==1; }

int main () {
  std::vector<int> foo {1,2,3,4,5,6,7,8,9}; // 1 2 3 4 5 6 7 8 9

  std::partition (foo.begin(),foo.end(),IsOdd);
  auto it = std::is_sorted_until(foo.begin(),foo.end());

  std::cout << "sorted elements:";
  for (std::vector<int>::iterator i=foo.begin(); i!=it; ++i)
    std::cout << ' ' << *it;
  std::cout << '\n';

  std::cout << "unsorted elements:";
  for (std::vector<int>::iterator i=it; i!=foo.end(); ++i)
    std::cout << ' ' << *it;
  std::cout << '\n';

  return 0;
}
