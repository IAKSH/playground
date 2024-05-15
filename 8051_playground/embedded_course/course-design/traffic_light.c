#include <reg51.h>

#define S7D2_DATA_PORT (P2)
#define S7D2_CS_PORT (P0)
#define LED_PORT (P1)

const unsigned char S7D2_DATA_MAP[10] = {
	0x3f,0x06,0x5b,0x4f,0x66,0x6d,0x7d,0x07,0x7f,0x6f};

char ew_duration[3] = {30,30,3};// {30,30,3}
char sn_duration[3] = {30,30,3};

char ew_current_state = 0;// 0,1,2: red,green,yellow
char ew_current_duration;
char sn_current_state = 1;// 0,1,2: red,green,yellow
char sn_current_duration;

// 12MHz 16bit 50ms
#define TH0_VAL (0x3c)
#define TL0_VAL (0xb0)
char t0_cycle = 0;

void reload_timer0() {
	TH0 = TH0_VAL;
	TL0 = TL0_VAL;
}

void init_t0() {
	TMOD = 0x01;
	reload_timer0();
	TR0 = 1;
	ET0 = 1;
}

void enable_interrupt() {
	EA = 1;
}

void init_data() {
	LED_PORT = 0x2e;
	//LED_PORT = 0x1b;
	ew_current_duration = ew_duration[ew_current_state];
	sn_current_duration = sn_duration[sn_current_state];
}

sbit LED_PIN_0 = P1^0;
sbit LED_PIN_1 = P1^1;
sbit LED_PIN_2 = P1^2;
sbit LED_PIN_3 = P1^3;
sbit LED_PIN_4 = P1^4;
sbit LED_PIN_5 = P1^5;

void ew_led_update() {
	//LED_PORT |= 0x07;// fill 1 into low 3bit
	//LED_PORT ^= (0x01 << ew_current_state);// flip a single bit
	LED_PIN_0 = 1;
	LED_PIN_1 = 1;
	LED_PIN_2 = 1;
	switch(ew_current_state) {
		case 0:
			LED_PIN_0 = 0;
			break;
		case 1:
			LED_PIN_1 = 0;
			break;
		case 2:
			LED_PIN_2 = 0;
			break;
	}
}

sbit P1_6 = P1^6;

void sn_led_update() {
	//LED_PORT |= 0x38;// fill 1 into low 3bit, with 3bit offset
	//LED_PORT ^= (0x08 << ew_current_state);// flip a single bit, with 3bit offset
	LED_PIN_3 = 1;
	LED_PIN_4 = 1;
	LED_PIN_5 = 1;
	switch(sn_current_state) {
		case 0:
			LED_PIN_3 = 0;
			break;
		case 1:
			LED_PIN_4 = 0;
			break;
		case 2:
			//P1_6 = !P1_6;// cant get here!
			LED_PIN_5 = 0;
			break;
	}
}

void update_ew_state() {
	if(--ew_current_duration == -1) {
		P0 = !P0;
		if((ew_current_state == 0 && (sn_current_state == 1 || (sn_current_state == 2 && sn_current_duration != 0)))) {
			++ew_current_duration;
			return;
		}
		ew_current_state = ++ew_current_state % 3;
		ew_current_duration = ew_duration[ew_current_state];
		ew_led_update();
	}
}

void update_sn_state() {
	if(--sn_current_duration == -1) {
		// can't add && ew_current_duration != 0 here for some reason that I don't know yet
		// or it will cause sync fall at every second yellow light
		if(sn_current_state == 0 && (ew_current_state == 1 || ew_current_state == 2)) {
			++sn_current_duration;
			return;
		}
		sn_current_state = ++sn_current_state % 3;
		sn_current_duration = sn_duration[sn_current_state];
		sn_led_update();
	}
}

void init_button_ite() {
	EX0 = 1;
	IT0 = 1;
	EX1 = 1;
	IT1 = 1;
}

void t0_irs() interrupt 1 {
	reload_timer0();
	if(++t0_cycle == 20) {
		t0_cycle = 0;
		update_ew_state();
		update_sn_state();
	}
}

void et0_irs() interrupt 0 {
	ew_duration[0] = 40;
	ew_duration[1] = 20;
	sn_duration[0] = 20;
	sn_duration[1] = 40;
}

void et1_irs() interrupt 2 {
	ew_duration[0] = 20;
	ew_duration[1] = 40;
	sn_duration[0] = 40;
	sn_duration[1] = 20;
}

//12MHz
void delay10ms() {
	unsigned char i, j;

	i = 20;
	j = 113;
	do {
		while (--j);
	} while (--i);
}

void update_ew_s7d2() {
	S7D2_CS_PORT = 0xfd;
	S7D2_DATA_PORT = S7D2_DATA_MAP[ew_current_duration % 10];
	delay10ms();
	S7D2_CS_PORT = 0xfe;
	S7D2_DATA_PORT = S7D2_DATA_MAP[ew_current_duration / 10];
	delay10ms();
}

void update_sn_s7d2() {
	S7D2_CS_PORT = 0xf7;
	S7D2_DATA_PORT = S7D2_DATA_MAP[sn_current_duration % 10];
	delay10ms();
	S7D2_CS_PORT = 0xfb;
	S7D2_DATA_PORT = S7D2_DATA_MAP[sn_current_duration / 10];
	delay10ms();
}

void main() {
	init_data();
	init_t0();
	init_button_ite();
	enable_interrupt();
	while(1) {
		update_ew_s7d2();
		update_sn_s7d2();
	}
}