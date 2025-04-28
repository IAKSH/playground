#include <reg51.h>

char led1_codes[] = {0x4f,0x5b};
char led2_codes[] = {0x76,0x38};
bit i = 1;

void int0Srv() interrupt 0 {
	i = 1;
}

void int1Srv() interrupt 2 {
	i = 0;
}

void delay1ms()		//@12.000MHz
{
	unsigned char i, j;

	i = 2;
	j = 239;
	do
	{
		while (--j);
	} while (--i);
}

int main() {
	P2 = 0x00;
	
	EA = 1;
	EX0 = 1;
	IT0 = 1;
	EX1 = 1;
	IT1 = 1;
	
	while(1) {		
		P1 = 0x01;
		P2 = led1_codes[i];
		delay1ms();
		P1 = 0x02;
		P2 = led2_codes[i];
		delay1ms();
	}
}