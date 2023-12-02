#include <iostream>
#include <memory>
#include <deque>

struct Data
{
    int val;
    Data(int i)
        : val(i)
    {}
    ~Data(){std::cout << "~Data()\n";}
};

int main()
{
    std::deque<std::unique_ptr<Data>> datas;
    for(auto& item : {114,514,1919,810})
        datas.push_back(std::make_unique<Data>(item));

    for(auto& item : datas)
        std::cout << item << ':' << item->val << '\t';
    std::cout << std::endl;

    std::unique_ptr<Data> data = std::move(datas.at(1));
    std::cout << "moved out " << data << ':' << data->val << std::endl;

    for(auto& item : datas)
    {
        if(item)
            std::cout << item << ':' << item->val << '\t';
        else
            std::cout << item << ":null\t";
    }

    std::cout << std::endl;
}