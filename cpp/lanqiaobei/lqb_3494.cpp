// https://www.lanqiao.cn/problems/3494/learning/?page=1&first_category_id=1&second_category_id=3&tags=2023

#include <bits/stdc++.h>

using namespace std;

// 需要再熟悉下使用std::chrono在日期时间字符串，时间戳和各种时分秒毫秒微妙之间的转换

int main() {
    vector<int64_t> v;

    ifstream ifs("lqb_3494.in",ios::in);
    while(!ifs.eof()) {
        std::string s;
        getline(ifs,s);
        std::istringstream ss(s);
        std::tm tm = {};
        ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");
        auto tp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
        auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(tp.time_since_epoch()).count();
        v.emplace_back(timestamp);
    }
    ifs.close();

    sort(v.begin(),v.end(),less<int64_t>());
    
    int len = v.size();
    int work_time = 0;
    for(int i = 0;i < len;i += 2)
        work_time += chrono::seconds(v[i + 1] - v[i]).count();

    cout << work_time << '\n';
    return 0;
}