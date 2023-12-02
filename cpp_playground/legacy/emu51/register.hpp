#pragma once

#include <cinttypes>
#include <iostream>
#include <array>

namespace v51
{
    template <size_t bits>
    class Register
    {
        static_assert(bits % 8 == 0, "bits must be a multiple of 8");

    private:
        std::array<uint8_t*,bits / 8> data;

    public:
        Register(uint8_t& head)
        {
            size_t bytes = bits / 8;
            for(int i = 0;i < bytes;i++)
                data[i] = &head + i;
        }
        ~Register() = default;

        void set(size_t n,bool b)
        {
            try
            {
                uint8_t& byte = *data.at(n / 8);
                if(b)
                    byte |= (1l << n);
                else
                    byte &= ~(1l << n);
            }
            catch(std::exception& e)
            {
                std::cerr << "exception: " << e.what() << std::endl;
                std::terminate();
            }
        }

        void setByte(size_t n,uint8_t b)
        {
            try
            {
                uint8_t& byte = *data.at(n);
                byte = b;
            }
            catch(std::exception& e)
            {
                std::cerr << "exception: " << e.what() << std::endl;
                std::terminate();
            }
        }

        bool get(size_t n) const
        {
            try
            {
                uint8_t& byte = *data.at(n / 8);
                return (byte >> n & 1) == 1;
            }
            catch(std::exception& e)
            {
                std::cerr << "exception: " << e.what() << std::endl;
                std::terminate();
            }
        }

        bool getByte(size_t n) const
        {
            try
            {
                return *data.at(n / 8);
            }
            catch(std::exception& e)
            {
                std::cerr << "exception: " << e.what() << std::endl;
                std::terminate();
            }
        }
    };
}