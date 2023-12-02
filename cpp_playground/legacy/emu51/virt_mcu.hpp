#pragma once

#include "register.hpp"
#include "program.hpp"
#include <cinttypes>
#include <exception>
#include <iostream>
#include <memory>
#include <array>

namespace v51
{
    class VirtMCU
    {
    private:
        const size_t ramMaxSize;
        const size_t romMaxSize;
        std::unique_ptr<unsigned char[]> rom;
        std::unique_ptr<unsigned char[]> ram;
        Register<8> acc{ ram[0xe0] };
        Register<16> dptr{ ram[0x82] };
        Register<8> dph{ ram[0x83] };
        Register<8> dpl{ ram[0x82] };
        Register<8> b{ ram[0xf0] };
        Register<8> sp{ ram[0x81] };
        Register<8> psw{ ram[0xd0] };
        Register<8> p0{ ram[0x80] };
        Register<8> p1{ ram[0x90] };
        Register<8> p2{ ram[0xa0] };
        Register<8> p3{ ram[0xb0] };
        Register<8> sbuf{ ram[0x99] };
        Register<8> ie{ ram[0xa8] };
        Register<8> scon{ ram[0x98] };
        Register<8> th1{ ram[0x8d] };
        Register<8> th0{ ram[0x8c] };
        Register<8> tl1{ ram[0x8b] };
        Register<8> tl0{ ram[0x8a] };
        Register<8> tmod{ ram[0x89] };
        Register<8> tcon{ ram[0x88] };
        Register<8> pcon{ ram[0x87] };
        Register<8> r{ ram[0x00] };
        uint16_t pc { 0x0000 };

        void initialize();

    public:
        VirtMCU(size_t romSize,size_t ramSize);
        ~VirtMCU() = default;

        void process(Program program,size_t line);
    };
}