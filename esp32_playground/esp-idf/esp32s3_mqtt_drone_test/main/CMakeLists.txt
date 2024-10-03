idf_component_register(SRCS 
                        "app_main.c"
                        "drone_status.c"
                        "./wireless/mqtt_trans.c"
                        "./task/posture/posture.c"
                        "./task/stabilizer/stabilizer.c"
                    INCLUDE_DIRS "."
                    REQUIRES mpu6050 bmp280 motor nvs_flash esp_event esp_netif protocol_examples_common mqtt driver)