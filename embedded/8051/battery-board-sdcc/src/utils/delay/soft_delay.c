#include "soft_delay.h"

#define _nop_()  __asm nop __endasm;

// STC-Y6 @24.000MHz
void soft_delay_1sec(void) {
	unsigned char i, j, k;
	_nop_();
	_nop_();
	i = 122;
	j = 193;
	k = 128;
	do {
		do {
			while (--k);
		} while (--j);
	} while (--i);
}

// STC-Y6 @24.000MHz
void soft_delay_1ms(void) {
	unsigned char i, j;
	_nop_();
	i = 32;
	j = 40;
	do {
		while (--j);
	} while (--i);
}

// STC-Y6 @24.000MHz
void soft_delay_10ms(void) {
	unsigned char i, j, k;
	i = 2;
	j = 56;
	k = 172;
	do {
		do {
			while (--k);
		} while (--j);
	} while (--i);
}

// STC-Y6 @24.000MHz
void soft_delay_100ms(void) {
	unsigned char i, j, k;
	_nop_();
	_nop_();
	i = 13;
	j = 45;
	k = 214;
	do {
		do {
			while (--k);
		} while (--j);
	} while (--i);
}

// STC-Y6 @24.000MHz
void soft_delay_1us(void) {
	unsigned char i;
	i = 6;
	while (--i);
}


// STC-Y6 @24.000MHz
void soft_delay_10us(void) {
	unsigned char i;
	i = 78;
	while (--i);
}

// STC-Y6 @24.000MHz
void soft_delay_100us(void) {
	unsigned char i, j;
	i = 4;
	j = 27;
	do {
		while (--j);
	} while (--i);
}
