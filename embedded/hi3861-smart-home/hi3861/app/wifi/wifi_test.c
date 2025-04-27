#include <stdio.h>
#include "ohos_init.h"
#include "ohos_types.h"
#include "cmsis_os2.h"
#include "wifi_device.h"
#include "lwip/netifapi.h"
#include "lwip/api_shell.h"

#include "tcp_client.h"
#include "tcp_server.h"

#define STACK_SIZE     10240
#define AP_SSID        "Lo-Fi"
#define AP_SKEY        "HelloNavi"

#define IDX_0          0
#define IDX_1          1
#define IDX_2          2
#define IDX_3          3
#define IDX_4          4
#define IDX_5          5


static void PrintLinkedInfo(WifiLinkedInfo* info)
{
    if (!info) return;

    static char macAddress[32] = {0};
    unsigned char* mac = info->bssid;
    // int ret = snprintf(macAddress, sizeof(macAddress), "%02X:%02X:%02X:%02X:%02X:%02X",
    int ret = snprintf_s(macAddress, sizeof(macAddress), sizeof(macAddress) - 1, "%02X:%02X:%02X:%02X:%02X:%02X",
        mac[IDX_0], mac[IDX_1], mac[IDX_2], mac[IDX_3], mac[IDX_4], mac[IDX_5]);
    if (ret < 0) {
        return;
    }
    printf("[wifi] bssid: %s, rssi: %d, connState: %d, reason: %d, ssid: %s\r\n",
        macAddress, info->rssi, info->connState, info->disconnectedReason, info->ssid);
}

int wifi_connected = 0;

static void on_wifi_connection_changed(int state, WifiLinkedInfo* info) {
    if(!info) {
        return;
    }

    printf("[wifi] %s %d, staate = %d\n",__FUNCTION__, __LINE__, state);
    PrintLinkedInfo(info);

    if (state == WIFI_STATE_AVALIABLE) {
        wifi_connected = 1;
    } else {
        wifi_connected = 0;
    }
}

static void on_wifi_scan_state_changed(int state,int size) {
    printf("[wifi] %s %d, state = %X, size = %d\n",__FUNCTION__, __LINE__, state, size);
} 

static void wifi_task(void) {
    WifiErrorCode err_code;
    WifiEvent event_listener = {
        .OnWifiConnectionChanged = on_wifi_connection_changed,
        .OnWifiScanStateChanged = on_wifi_scan_state_changed
    };
    WifiDeviceConfig ap_config = {};
    int net_id = -1;

    osDelay(10);
    err_code = RegisterWifiEvent(&event_listener);
    printf("[wifi] RegisterWifiEvent: %d\n",err_code);

    strcpy(ap_config.ssid,AP_SSID);
    strcpy(ap_config.preSharedKey,AP_SKEY);
    ap_config.securityType = WIFI_SEC_TYPE_PSK;

    // Non-blocking Wi-Fi connection with state machine
    enum WifiState { INIT, ENABLE_WIFI, CONFIGURE, CONNECT, DHCP_START, CONNECTED, DISCONNECT } state = INIT;
    struct netif* iface = NULL;

    while (1) {
        switch (state) {
            case INIT:
                err_code = RegisterWifiEvent(&event_listener);
                printf("[wifi] RegisterWifiEvent: %d\n", err_code);
                state = ENABLE_WIFI;
                break;
            case ENABLE_WIFI:
                err_code = EnableWifi();
                printf("[wifi] EnableWifi: %d\n", err_code);
                state = CONFIGURE;
                break;
            case CONFIGURE:
                err_code = AddDeviceConfig(&ap_config, &net_id);
                printf("[wifi] AddDeviceConfig: %d\n", err_code);
                state = CONNECT;
                break;
            case CONNECT:
                wifi_connected = 0;
                err_code = ConnectTo(net_id);
                printf("[wifi] ConnectTo(%d): %d\n", net_id, err_code);
                state = DHCP_START;
                break;
            case DHCP_START:
                if (wifi_connected) {
                    iface = netifapi_netif_find("wlan0");
                    if (iface) {
                        err_code = netifapi_dhcp_start(iface);
                        printf("[wifi] netifapi_dhcp_start: %d\n", err_code);
                        state = CONNECTED;
                    }
                }
                break;
            case CONNECTED:
                // Perform network tasks here
                printf("[wifi] connected: %d\n", wifi_connected);
                // Perform some operations then disconnect
                tcp_server_startup();
                state = DISCONNECT;
                break;
            case DISCONNECT:
                if (iface) {
                    err_code = netifapi_dhcp_stop(iface);
                    printf("[wifi] netifapi_dhcp_stop: %d\n", err_code);
                }
                Disconnect();
                RemoveDevice(net_id);
                err_code = DisableWifi();
                printf("[wifi] DisableWifi: %d\n", err_code);
                state = INIT;
                break;
        }
        osDelay(10); // Reduce delay to allow more frequent task switching
        osThreadYield(); // Yield to allow other tasks to run
    }
}

static void wifi_test(void) {
    osThreadAttr_t attr;

    attr.name = "wifi_test";
    attr.attr_bits = 0U;
    attr.cb_mem = NULL;
    attr.cb_size = 0U;
    attr.stack_mem = NULL;
    attr.stack_size = STACK_SIZE;
    attr.priority = osPriorityBelowNormal;

    if (osThreadNew(wifi_task, NULL, &attr) == NULL) {
        printf("[wifi] Failed to create wifi task!\n");
    }
}

APP_FEATURE_INIT(wifi_test);
