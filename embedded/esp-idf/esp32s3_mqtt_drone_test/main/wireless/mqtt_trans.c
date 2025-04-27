#include "mqtt_trans.h"
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "esp_system.h"
#include "nvs_flash.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "protocol_examples_common.h"
#include "esp_log.h"
#include "mqtt_client.h"
#include "../drone_status.h"
#include <motor.h>

static const char *TAG = "mqtt5_example";

static void log_error_if_nonzero(const char *message, int error_code) {
    if (error_code != 0) {
        ESP_LOGE(TAG, "Last error %s: 0x%x", message, error_code);
    }
}

static int msg_id_gryo_euler, msg_id_gryo_accel, msg_id_gryo_temperature,
    msg_id_barometer_pressure, msg_id_barometer_temperature, msg_id_barometer_altitude, msg_id_motor_duty;
static esp_mqtt_client_handle_t client;

static void task_mqtt_publish() {
    char mqtt_str[40];
    while (true) {
        sprintf(mqtt_str, "%.2f,%.2f,%.2f", drone_gryo_euler[0], drone_gryo_euler[1], drone_gryo_euler[2]);
        msg_id_gryo_euler = esp_mqtt_client_publish(client, "/drone/gryo/euler", mqtt_str, 0, 1, 1);

        sprintf(mqtt_str, "%.2f,%.2f,%.2f", drone_gryo_accel[0], drone_gryo_accel[1], drone_gryo_accel[2]);
        msg_id_gryo_accel = esp_mqtt_client_publish(client, "/drone/gryo/accel", mqtt_str, 0, 1, 1);

        sprintf(mqtt_str, "%.2f", drone_gyro_temperature);
        msg_id_gryo_temperature = esp_mqtt_client_publish(client, "/drone/gryo/temperature", mqtt_str, 0, 1, 1);

        sprintf(mqtt_str, "%.2f", drone_barometer_pressure);
        msg_id_barometer_pressure = esp_mqtt_client_publish(client, "/drone/barometer/pressure", mqtt_str, 0, 1, 1);

        sprintf(mqtt_str, "%.2f", drone_barometer_temperature);
        msg_id_barometer_temperature = esp_mqtt_client_publish(client, "/drone/barometer/temperature", mqtt_str, 0, 1, 1);

        sprintf(mqtt_str, "%.2f", drone_barometer_altitude);
        msg_id_barometer_altitude = esp_mqtt_client_publish(client, "/drone/barometer/altitude", mqtt_str, 0, 1, 1);

        sprintf(mqtt_str, "%d,%d,%d,%d", drone_motor_duty[1], drone_motor_duty[2], drone_motor_duty[3], drone_motor_duty[4]);
        msg_id_motor_duty = esp_mqtt_client_publish(client, "/drone/motor/duty", mqtt_str, 0, 1, 1);

        vTaskDelay(pdMS_TO_TICKS(50));
    }
}

// New function to handle received data
static void handle_incoming_data(esp_mqtt_event_handle_t event) {
    char data[20];
    snprintf(data, event->data_len + 1, "%.*s", event->data_len, event->data);

    if (strncmp(event->topic, "/drone/control/motor_offset", event->topic_len) == 0) {
        drone_motor_offset = atoi(data);
    } else if (strncmp(event->topic, "/drone/control/kp", event->topic_len) == 0) {
        drone_motor_kp = atoi(data);
    } else if (strncmp(event->topic, "/drone/control/ki", event->topic_len) == 0) {
        drone_motor_ki = atoi(data);
    } else if (strncmp(event->topic, "/drone/control/kd", event->topic_len) == 0) {
        drone_motor_kd = atoi(data);
    } else if (strncmp(event->topic, "/drone/control/euler/roll", event->topic_len) == 0) {
        drone_target_euler[0] = atoi(data);
    } else if (strncmp(event->topic, "/drone/control/euler/pitch", event->topic_len) == 0) {
        drone_target_euler[1] = atoi(data);
    } else if (strncmp(event->topic, "/drone/control/euler/yaw", event->topic_len) == 0) {
        drone_target_euler[2] = atoi(data);
    } else if (strncmp(event->topic, "/drone/control/emergency_stop", event->topic_len) == 0) {
        drone_motor_emergency_stop = atoi(data);
    }
}

static void mqtt5_event_handler(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data) {
    ESP_LOGD(TAG, "Event dispatched from event loop base=%s, event_id=%" PRIi32, base, event_id);
    esp_mqtt_event_handle_t event = event_data;
    client = event->client;

    ESP_LOGD(TAG, "free heap size is %" PRIu32 ", minimum %" PRIu32, esp_get_free_heap_size(), esp_get_minimum_free_heap_size());

    switch ((esp_mqtt_event_id_t)event_id) {
        case MQTT_EVENT_CONNECTED:
            esp_mqtt_client_subscribe(client, "/drone/control/motor_offset", 1);
            esp_mqtt_client_subscribe(client, "/drone/control/kp", 1);
            esp_mqtt_client_subscribe(client, "/drone/control/ki", 1);
            esp_mqtt_client_subscribe(client, "/drone/control/kd", 1);
            esp_mqtt_client_subscribe(client, "/drone/control/euler/roll", 1);
            esp_mqtt_client_subscribe(client, "/drone/control/euler/pitch", 1);
            esp_mqtt_client_subscribe(client, "/drone/control/euler/yaw", 1);
            esp_mqtt_client_subscribe(client, "/drone/control/emergency_stop", 1);
            xTaskCreate(task_mqtt_publish, "mqtt_publish", 4096, NULL, 5, NULL);
            break;
        case MQTT_EVENT_DISCONNECTED:
            break;
        case MQTT_EVENT_SUBSCRIBED:
            break;
        case MQTT_EVENT_UNSUBSCRIBED:
            break;
        case MQTT_EVENT_PUBLISHED:
            break;
        case MQTT_EVENT_DATA:
            handle_incoming_data(event);
            break;
        case MQTT_EVENT_ERROR:
            ESP_LOGI(TAG, "MQTT_EVENT_ERROR");
            ESP_LOGI(TAG, "MQTT5 return code is %d", event->error_handle->connect_return_code);
            if (event->error_handle->error_type == MQTT_ERROR_TYPE_TCP_TRANSPORT) {
                log_error_if_nonzero("reported from esp-tls", event->error_handle->esp_tls_last_esp_err);
                log_error_if_nonzero("reported from tls stack", event->error_handle->esp_tls_stack_err);
                log_error_if_nonzero("captured as transport's socket errno", event->error_handle->esp_transport_sock_errno);
                ESP_LOGI(TAG, "Last errno string (%s)", strerror(event->error_handle->esp_transport_sock_errno));
            }
            break;
        default:
            ESP_LOGI(TAG, "Other event id:%d", event->event_id);
            break;
    }
}

void mqtt_startup(void) {
    esp_log_level_set("*", ESP_LOG_WARN);
    esp_log_level_set("mqtt_client", ESP_LOG_WARN);
    esp_log_level_set("mqtt_example", ESP_LOG_WARN);
    esp_log_level_set("transport_base", ESP_LOG_WARN);
    esp_log_level_set("esp-tls", ESP_LOG_WARN);
    esp_log_level_set("transport", ESP_LOG_WARN);
    esp_log_level_set("outbox", ESP_LOG_WARN);
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    /* This helper function configures Wi-Fi or Ethernet, as selected in menuconfig.
     * Read "Establishing Wi-Fi or Ethernet Connection" section in
     * examples/protocols/README.md for more information about this function.
     */
    ESP_ERROR_CHECK(example_connect());
    esp_mqtt5_connection_property_config_t connect_property = {
        .session_expiry_interval = 10,
        .maximum_packet_size = 1024,
        .receive_maximum = 65535,
        .topic_alias_maximum = 2,
        .request_resp_info = true,
        .request_problem_info = true,
        .will_delay_interval = 10,
        .payload_format_indicator = true,
        .message_expiry_interval = 10,
        .response_topic = "/test/response",
        .correlation_data = "123456",
        .correlation_data_len = 6,
    };
    esp_mqtt_client_config_t mqtt5_cfg = {
        .broker.address.uri = CONFIG_BROKER_URL,
        .session.protocol_ver = MQTT_PROTOCOL_V_5,
        .network.disable_auto_reconnect = true,
        //.credentials.username = "123",
        //.credentials.authentication.password = "456",
        .session.last_will.topic = "/topic/will",
        .session.last_will.msg = "i will leave",
        .session.last_will.msg_len = 12,
        .session.last_will.qos = 1,
        .session.last_will.retain = true,
    };
    client = esp_mqtt_client_init(&mqtt5_cfg);
    /* Set connection properties and user properties */
    esp_mqtt5_client_set_connect_property(client, &connect_property);
    /* If you call esp_mqtt5_client_set_user_property to set user properties, DO NOT forget to delete them.
     * esp_mqtt5_client_set_connect_property will malloc buffer to store the user_property and you can delete it after
     */
    esp_mqtt5_client_delete_user_property(connect_property.user_property);
    esp_mqtt5_client_delete_user_property(connect_property.will_user_property);
    /* The last argument may be used to pass data to the event handler, in this example mqtt_event_handler */
    esp_mqtt_client_register_event(client, ESP_EVENT_ANY_ID, mqtt5_event_handler, NULL);
    esp_mqtt_client_start(client);
}
