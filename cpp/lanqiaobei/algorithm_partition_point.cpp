#include <iostream> // std::cout
#include <algorithm> // std::partition, std::partition_point
#include <vector> // std::vector

/*
 * std::partition_point
 * 用于查找已经被分区（如通过 std::partition）的范围 [first, last) 中的分区点 
 */

bool IsOdd (int i) { return (i%2)==1; }

int main () {
  std::vector<int> foo {1,2,3,4,5,6,7,8,9}; // 1 2 3 4 5 6 7 8 9

  std::partition (foo.begin(),foo.end(),IsOdd);
  auto it = std::partition_point (foo.begin(),foo.end(),IsOdd);

  std::cout << "odd elements:";
  for (std::vector<int>::iterator i=foo.begin(); i!=it; ++i)
    std::cout << ' ' << *i;
  std::cout << '\n';

  std::cout << "even elements:";
  for (std::vector<int>::iterator i=it; i!=foo.end(); ++i)
    std::cout << ' ' << *i;
  std::cout << '\n';

  return 0;
}
