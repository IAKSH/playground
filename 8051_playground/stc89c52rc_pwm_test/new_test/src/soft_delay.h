#pragma once

//@11.0592MHz
inline static void delay_1ms()
{
	unsigned char i, j;

	__asm
    NOP
    __endasm;

	i = 11;
	j = 190;
	do
	{
		while (--j);
	} while (--i);
}