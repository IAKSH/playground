#include "bflb_mtimer.h"
#include "bflb_uart.h"
#include "board.h"
  
struct bflb_device_s *uartx;

int main(void)
{
    board_init();
    board_uartx_gpio_init();
  
    uartx = bflb_device_get_by_name(DEFAULT_TEST_UART);
  
    struct bflb_uart_config_s cfg;
  
    // for BL616
    // UART1 TX = GPIO_23
    // UART1 RX = GPIO_24
  
    cfg.baudrate = 115200;
    cfg.data_bits = UART_DATA_BITS_8;
    cfg.stop_bits = UART_STOP_BITS_1;
    cfg.parity = UART_PARITY_NONE;
    cfg.flow_ctrl = 0;
    cfg.tx_fifo_threshold = 7;
    cfg.rx_fifo_threshold = 7;
    bflb_uart_init(uartx, &cfg);
  
    char ret_str[] = "Hello, your input is ";
    char ch;
    
    while (1) {
        ch = bflb_uart_getchar(uartx);
        if ((ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')) {
            for(int i = 0;i < sizeof(ret_str);i++)
                bflb_uart_putchar(uartx, ret_str[i]);
            bflb_uart_putchar(uartx, ch);
        }
    }
}