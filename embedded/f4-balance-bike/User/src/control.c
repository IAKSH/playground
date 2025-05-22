#include "driver/nrf24l01p.h"
#include "command.h"
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define COMMAND_FRAG_NUM_MAX 128

static uint8_t rx_address[5] = {0x0, 0x0, 0x0, 0x0, 0x01};

static const char* command_type_to_str(CommandType type) {
    switch(type) {
        case COMMAND_MOVE:
            return "move";
        case COMMAND_VOLT:
            return "volt";
        case COMMAND_CAM_ROTATE:
            return "cam_rotate";
        case COMMAND_CAM_SHOT:
            return "cam_shot";
        case COMMAND_PID:
            return "pid";
        default:
            return "unknown";
    }
}

static void command_move_handler(CommandPacket command) {
    printf("speed: {%d, %d}\n", command.payload.move.speed[0], command.payload.move.speed[1]);
}

void control_task(void* arg) {
    nrf24l01p_rx_init(2500, _1Mbps);

    if (!nrf24l01p_check()) {
        printf("nrf24l01+ no response\n");
        Error_Handler();
    }

    nrf24l01p_set_rx_addr(0,rx_address,5);

    CommandPacket command;
    CommandFrag command_frags[COMMAND_FRAG_NUM_MAX];
    uint16_t command_frag_cnt = 0;

    const uint32_t frag_timeout_ms = 100;
    osStatus_t status;

    while (1) {
        status = osMessageQueueGet(command_queue, &command_frags[command_frag_cnt], NULL, frag_timeout_ms);
        
        if (status == osErrorTimeout) {
            // 超时未收到分包，认为此次命令传输中断，清空当前缓冲区
            if (command_frag_cnt > 0) {
                printf("Timeout: incomplete command packet discarded (received %d fragments)\n", command_frag_cnt);
                command_frag_cnt = 0;
            }
            // 继续等待新的分包数据
            continue;
        }
        
        // 如果当前分包设置了end标志，则收到了完整命令
        if (command_frags[command_frag_cnt].end) {
            int total_frags = command_frag_cnt + 1;  // 包含当前分包
            uint8_t *p = (uint8_t *)&command;
            const size_t frag_payload_size = sizeof(command_frags[0].payload);
            
            // 将所有分包 payload 数据依次复制到 command 内存中
            for (int i = 0; i < total_frags; i++) {
                size_t copy_len = frag_payload_size;
                // 最后一个分包可能不足一个完整 payload 大小
                if (i == total_frags - 1) {
                    size_t remaining = sizeof(CommandPacket) - (p - (uint8_t *)&command);
                    copy_len = remaining;
                }
                memcpy(p, command_frags[i].payload, copy_len);
                p += copy_len;
            }

            printf("command received: type = %s\n", command_type_to_str(command.type));
            switch (command.type) {
                case COMMAND_MOVE:
                    command_move_handler(command);
                    break;
                default:
                    break;
            }
            // 重置分包计数器，准备接收下一条命令
            command_frag_cnt = 0;
        } else {
            // 未收到end标志则继续累计分包
            command_frag_cnt++;
            // 防止分包数量超限，必要时也可加判断，超过一定数量直接丢弃
            if (command_frag_cnt >= COMMAND_FRAG_NUM_MAX) {
                printf("Warning: fragment count exceeded. Discarding incomplete command.\n");
                command_frag_cnt = 0;
            }
        }
    }
}
