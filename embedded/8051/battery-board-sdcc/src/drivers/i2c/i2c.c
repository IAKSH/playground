#include <drivers/register/stc8g.h>
#include "i2c.h"
#include "i2c_config.h"

/*
 * 初始化 I2C 外设
 * - 配置对应引脚的模式为 I2C 模式
 */
void i2c_init(void) {
    P_SW2 = 0x80;
#if I2C_CHANNEL == 2
    // 关闭P_SW2中第5位，打开第4位，实现 I2C2 的引脚重映射
    P_SW2 &= ~(1 << 5);
    P_SW2 |= (1 << 4);
#endif
    I2CCFG = 0xe0;
    I2CMSST = 0x00;
}

/*
 * 等待 I2C 操作完成
 * 检查 I2CMSST 寄存器的 0x40 位是否置位，
 * 置位表示当前操作完成，随后清除该标志位。
 */
void i2c_wait(void) {
    while (!(I2CMSST & 0x40));  // 等待操作完成
    I2CMSST &= ~0x40;           // 清除完成标志
}

/*
 * 发送 I2C 起始信号
 * 将 I2CMSCR 置为 0x01 表示启动信号，然后等待完成
 */
void i2c_start(void) {
    I2CMSCR = 0x01;  // 设置起始信号命令
    i2c_wait();
}

/*
 * 发送一个字节数据
 * 参数:
 *   data -- 要发送的数据字节
 */
void i2c_send_data(uint8_t data) {
    I2CTXD = data;   // 将数据加载到发送寄存器
    I2CMSCR = 0x02;  // 设置发送命令
    i2c_wait();
}

/*
 * 读取发送后的 ACK
 * 通过设置命令 0x03 发起 ACK 读取，然后等待完成
 */
void i2c_recv_ack(void) {
    I2CMSCR = 0x03;  // 读取 ACK 命令
    i2c_wait();
}

/*
 * 接收一个字节数据
 * 通过设置命令 0x04 发起数据接收，等待完成后返回接收到的数据
 */
uint8_t i2c_recv_data(void) {
    I2CMSCR = 0x04;  // 接收数据命令
    i2c_wait();
    return I2CRXD;   // 返回接收数据寄存器中的数据
}

/*
 * 发送 ACK（确认信号）
 * 将 I2CMSST 寄存器置 0x00，然后设置命令 0x05 发送 ACK，并等待完成
 */
void i2c_send_ack(void) {
    I2CMSST = 0x00;  // 清除状态或设置 ACK 标志（具体含义参考芯片手册）
    I2CMSCR = 0x05;  // 发送 ACK 命令
    i2c_wait();
}

/*
 * 发送 NAK（非确认信号）
 * 将 I2CMSST 寄存器置 0x01，然后设置命令 0x05 发送 NAK，并等待完成
 */
void i2c_send_nak(void) {
    I2CMSST = 0x01;  // 设置某位以表示 NAK（具体参考手册）
    I2CMSCR = 0x05;  // 同样使用命令 0x05，但状态不同，表示发送 NAK
    i2c_wait();
}

/*
 * 发送 I2C 停止信号
 * 设置命令 0x06 发起停止信号，然后等待完成
 */
void i2c_stop(void) {
    I2CMSCR = 0x06;  // 停止信号命令
    i2c_wait();
}
