#include <reg51.h>
#include <intrins.h>

void delay500ms();

int main() {
	P0 = 0xfe;
	while(1) {
		delay500ms();
		P0 <<= 1;
		++P0;
		if(P0 == 0xff)
			P0 = 0xfe;
	}
}

void delay500ms()		//@12.000MHz
{
	unsigned char i, j, k;

	_nop_();
	i = 4;
	j = 205;
	k = 187;
	do
	{
		do
		{
			while (--k);
		} while (--j);
	} while (--i);
}