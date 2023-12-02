#pragma once

#include <string>
#include <memory>

namespace v51
{
    class Program
    {
    private:
        std::unique_ptr<unsigned char> binary;

    public:
        Program(std::string_view path);
        ~Program();
    };
}