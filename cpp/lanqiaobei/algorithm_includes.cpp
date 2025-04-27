#include <iostream>
#include <algorithm>
#include <vector>
#include <string>

void func(const std::string& s1,const std::string& s2) noexcept {
    std::cout
        << '\"' << s1 << '\"' << " includes \"" << s2 << "\" = "
        << (std::includes(s1.begin(),s1.end(),s2.begin(),s2.end()) ? "true" : "false") << std::endl;
}

int main() {
    std::string s1 = "nihaoma?";
    std::string s2(s1);
    std::reverse(s2.begin(),s2.end());
    std::string s3 = "nihao";

    func(s1,s3);
    func(s2,s3);

    return 0;
}
