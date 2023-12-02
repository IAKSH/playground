#include "virt_mcu.hpp"

v51::VirtMCU::VirtMCU(size_t romSize,size_t ramSize)
    : romMaxSize(romSize),ramMaxSize(ramSize),
    rom(std::make_unique<unsigned char[]>(romMaxSize)),
    ram(std::make_unique<unsigned char[]>(ramMaxSize))
{
    initialize();
}

void v51::VirtMCU::initialize()
{
    for(size_t i = 0;i < ramMaxSize;i++)
        ram[i] = 0xa5;

    p0.setByte(0,0xff);
    p1.setByte(0,0xff);
    p2.setByte(0,0xff);
    p3.setByte(0,0xff);
    sp.setByte(0,0x07);
}

void v51::VirtMCU::process(Program program,size_t line)
{
    
}