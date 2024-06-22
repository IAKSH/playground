#include <reg51.h>

sbit LED1 = P1^0;
sbit LED2 = P1^1;
sbit LED3 = P1^2;
sbit LED4 = P1^3;

unsigned char pwm[4] = {16,32,64,128};
unsigned char dir[4] = {0,0,0,0};

void init_timers() {
    TMOD = 0x21;

    // timer0
    TH0 = 0xFF;
    TL0 = 0xFF;
    ET0 = 1;

    // timer1
    TH1 = 0xFD; // Set Timer 1 initial value for 9600 baud rate
    TL1 = 0xFD; // Set Timer 1 initial value for 9600 baud rate
    ET1 = 0;
}

void timer0_isr() interrupt 1 {
    static unsigned char pwm_count = 0;
    TH0 = 0xFF;
    TL0 = 0xFF;
    pwm_count++;
    LED1 = (pwm_count < pwm[0]) ? 0 : 1;
    LED2 = (pwm_count < pwm[1]) ? 0 : 1;
    LED3 = (pwm_count < pwm[2]) ? 0 : 1;
    LED4 = (pwm_count < pwm[3]) ? 0 : 1;
}

void init_pwm() {
    TR0 = 1;
}

void init_uart() {
    SCON = 0x50;
    ES = 1;
    TR1 = 1;
    PS = 1;// 提高优先级，防止串口的Timer1被饿死进不了中断
}

void uart_send_char(char c) {
    SBUF = c;
    while(!TI);
    TI = 0;
}

void uart_send_str(char* str) {
    for(;*str != '\0';++str)
        uart_send_char(*str);
}

char tmp;

void send_acc_data() {
    uart_send_str("set ACC to \"");
    uart_send_char(SBUF);
    uart_send_str("\"\nP of ACC is: ");
    ACC = SBUF;
    tmp = P;
    uart_send_char('0' + tmp);
}

char recieve;

void uart_isr() interrupt 4 {
    if (RI) {
        RI = 0;
        recieve = SBUF;
        //uart_send_char('!');
        send_acc_data();
    }
}

void update_pwm() {
	unsigned char i;
	for(i = 0;i < 4;i++) {
		if(pwm[i] == 0 || pwm[i] == 255) {
            dir[i] = !dir[i];
            if(i == 0) {
                uart_send_str("dir[0] = ");
                uart_send_char('0' + dir[0]);
                //uart_send_char('\n');
            }
        }
		pwm[i] += (dir[i] ? 1 : -1);
	}
}

void main() {
    init_timers();
    init_uart();
    init_pwm();

    EA = 1;

    uart_send_str("hello stc89c52rc!");

    while(1) {
        update_pwm();
    }
}
