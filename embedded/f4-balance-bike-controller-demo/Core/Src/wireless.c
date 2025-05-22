#include "main.h"
#include "nrf24l01p.h"
#include "wireless.h"
#include <stdio.h>
#include <string.h>

static NRF24L01P_Fragment fragments[NRF24L01P_FRAGMENT_MAX_LEN];

void wireless_irq(void) {
    if(nrf24l01p_check_tx_mode()) {
        nrf24l01p_check_ack();
    }
    /*
    else {
        NRF24L01P_Fragment frag;
        // 循环读完RX FIFO
        while ((nrf24l01p_get_fifo_status() & 0x01) == 0) { // RX_EMPTY == 0
            nrf24l01p_read_rx_fifo((uint8_t*)&frag);
            osMessageQueuePut(command_queue, &frag, 0, 0);
        }
        nrf24l01p_clear_rx_dr();
    }
    */
}

void wireless_send(void* data,uint16_t len) {
    const uint16_t payload_size = sizeof(fragments[0].payload);
    uint16_t frag_cnt = len / payload_size + ((len % payload_size) ? 1 : 0);

    // 分包
    uint8_t* p = data;
    for(uint16_t i = 0;i < frag_cnt;i++) {
        fragments[i].end = (i == frag_cnt - 1);
        uint16_t copy_len = payload_size;
        if(i == frag_cnt - 1 && (len % payload_size))
            copy_len = len % payload_size;
        memcpy(fragments[i].payload,p,copy_len);
        p += copy_len;
    }

    // 发送
    for(uint16_t i = 0;i < frag_cnt;i++) {
        if(!nrf24l01p_send_fragment((uint8_t*)&fragments[i])) {
            --i;
            continue;
        }
    }
}

/*
void wireless_receive(void* data, uint16_t len) {
    uint16_t frag_cnt = 0;
    osStatus_t status;

    while (1) {
        status = osMessageQueueGet(command_queue, &fragments[frag_cnt], NULL, WIRELESS_TIMEOUT_MS);

        if (status == osErrorTimeout) {
            // 超时未收到分包，丢弃已收分片
            if(frag_cnt != 0) {
                printf("wireless timeout, discarded\n");
                frag_cnt = 0;
            }
            continue;
        }

        if (fragments[frag_cnt].end) {
            uint16_t total_frags = frag_cnt + 1;
            uint8_t* p = data;
            const uint16_t frag_payload_size = sizeof(fragments[0].payload);
            for (uint16_t i = 0; i < total_frags; i++) {
                uint16_t copy_len = frag_payload_size;
                if (i == total_frags - 1) {
                    size_t remaining = len - (p - (uint8_t*)data);
                    copy_len = remaining;
                }
                memcpy(p, fragments[i].payload, copy_len);
                p += copy_len;
            }
            frag_cnt = 0;
            return;
        } else {
            frag_cnt++;
            if (frag_cnt >= NRF24L01P_FRAGMENT_MAX_LEN) {
                // 分片数量超限，丢弃
                printf("wireless overflow, discarded\n");
                frag_cnt = 0;
            }
        }
    }
}
*/