#ifdef ENABLE_STDOUT_REDIRECT
#include "main.h"

extern UART_HandleTypeDef huart1;

#if defined(__GNUC__) // arm-none-eabi-gcc
int _write(int fd,char* ptr,int len) {
    HAL_UART_Transmit(&huart1,(uint8_t *)ptr,len,HAL_MAX_DELAY);
    return len;
}
// TODO: 疑似粘包
int _read(int fd, char* ptr, int len) {
    for(int i = 0; i < len; i++) {
        uint8_t ch;
        HAL_UART_Receive(&huart1, &ch, 1, HAL_MAX_DELAY);
        ptr[i] = ch;
        if(ch == '\r' || ch == '\n') {
            ptr[i] = '\n';
            HAL_UART_Transmit(&huart1, (uint8_t*)"\r\n", 2, HAL_MAX_DELAY); // 回显换行
            return i + 1;
        }
        HAL_UART_Transmit(&huart1, &ch, 1, HAL_MAX_DELAY); // 回显
    }
    return len;
}
#elif defined (__ICCARM__) // IAR
#include "LowLevelIOInterface.h"
size_t __write(int handle, const unsigned char * buffer, size_t size) {
    HAL_UART_Transmit(&huart1, (uint8_t *) buffer, size, HAL_MAX_DELAY);
    return size;
}
size_t __read(int handle, unsigned char * buffer, size_t size) {
    for(size_t i = 0; i < size; i++) {
        uint8_t ch;
        HAL_UART_Receive(&huart1, &ch, 1, HAL_MAX_DELAY);
        buffer[i] = ch;
        if(ch == '\r' || ch == '\n') {
            buffer[i] = '\n';
            HAL_UART_Transmit(&huart1, (uint8_t*)"\r\n", 2, HAL_MAX_DELAY);
            return i + 1;
        }
        HAL_UART_Transmit(&huart1, &ch, 1, HAL_MAX_DELAY);
    }
    return size;
}
#elif defined (__CC_ARM) // Keil
int fputc(int ch, FILE *f) {
    HAL_UART_Transmit(&huart1, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}
int fgetc(FILE *f) {
    uint8_t ch = 0;
    HAL_UART_Receive(&huart1, &ch, 1, HAL_MAX_DELAY);
    if(ch == '\r' || ch == '\n') {
        HAL_UART_Transmit(&huart1, (uint8_t*)"\r\n", 2, HAL_MAX_DELAY);
        return '\n';
    }
    HAL_UART_Transmit(&huart1, &ch, 1, HAL_MAX_DELAY);
    return ch;
}
#endif
#endif