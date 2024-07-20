// Copyright Takatoshi Kondo 2019
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <iostream>
#include <iomanip>
#include <map>

#include <mqtt_client_cpp.hpp>
#include <sstream>

#define BROKER_HOST "192.168.1.110"
#define BROKER_PORT "1883"
#define CLIENT_ID "sensor"

extern float euler[3];

void mqtt_main() {
    MQTT_NS::setup_log();

    boost::asio::io_context ioc;

    std::uint16_t pid_sub1,pid_sub2;

    //int count = 0;
    // Create no TLS client
    // You can set the protocol_version to connect. If you don't set it, v3_1_1 is used.
    auto c = MQTT_NS::make_sync_client(ioc, BROKER_HOST, BROKER_PORT, MQTT_NS::protocol_version::v5);
    using packet_id_t = typename std::remove_reference_t<decltype(*c)>::packet_id_t;

    //auto disconnect = [&] {
    //    if (++count == 5) c->disconnect();
    //};

    // Setup client
    c->set_client_id(CLIENT_ID);
    c->set_clean_start(true);

    // Setup handlers

    // 连接确认
    // 在连接时进行订阅，Broker确认订阅后client跳入suback
    c->set_v5_connack_handler( // use v5 handler
        [&c, &pid_sub1, &pid_sub2]
        (bool sp, MQTT_NS::v5::connect_reason_code reason_code, MQTT_NS::v5::properties /*props*/){
            std::cout << "[client] Connack handler called" << std::endl;
            std::cout << "[client] Session Present: " << std::boolalpha << sp << std::endl;
            std::cout << "[client] Connect Reason Code: " << reason_code << std::endl;
            if (reason_code == MQTT_NS::v5::connect_reason_code::success) {
                pid_sub1 = c->subscribe("mqtt_client_cpp/temperature", MQTT_NS::qos::at_most_once);
                pid_sub2 = c->subscribe("/drone/gryo/euler", MQTT_NS::qos::at_least_once);
                //pid_sub2 = c->subscribe(
                //    std::vector<std::tuple<MQTT_NS::string_view, MQTT_NS::subscribe_options>>
                //    {
                //        { "mqtt_client_cpp/topic2_1", MQTT_NS::qos::at_least_once },
                //        { "mqtt_client_cpp/topic2_2", MQTT_NS::qos::exactly_once }
                //    }
                //);
            }
            return true;
        });

    c->set_close_handler( // this handler doesn't depend on MQTT protocol version
        []
        (){
            std::cout << "[client] closed." << std::endl;
        });

    
    c->set_error_handler( // this handler doesn't depend on MQTT protocol version
        []
        (MQTT_NS::error_code ec){
            std::cout << "[client] error: " << ec.message() << std::endl;
        });
    
    // 发布确认
    c->set_v5_puback_handler( // use v5 handler
        [&]
        (packet_id_t packet_id, MQTT_NS::v5::puback_reason_code reason_code, MQTT_NS::v5::properties /*props*/){
            std::cout <<
                "[client] puback received. packet_id: " << packet_id <<
                " reason_code: " << reason_code << std::endl;
            //disconnect();
            return true;
        });
    
    // 发布接受
    c->set_v5_pubrec_handler( // use v5 handler
        [&]
        (packet_id_t packet_id, MQTT_NS::v5::pubrec_reason_code reason_code, MQTT_NS::v5::properties /*props*/){
            std::cout <<
                "[client] pubrec received. packet_id: " << packet_id <<
                " reason_code: " << reason_code << std::endl;
            return true;
        });

    // 发布完成
    c->set_v5_pubcomp_handler( // use v5 handler
        [&]
        (packet_id_t packet_id, MQTT_NS::v5::pubcomp_reason_code reason_code, MQTT_NS::v5::properties /*props*/){
            std::cout <<
                "[client] pubcomp received. packet_id: " << packet_id <<
                " reason_code: " << reason_code << std::endl;
            //disconnect();
            return true;
        });

    // 订阅确认
    // 在Broker确认订阅后发布对应内容
    c->set_v5_suback_handler( // use v5 handler
        [&]
        (packet_id_t packet_id,
         std::vector<MQTT_NS::v5::suback_reason_code> reasons,
         MQTT_NS::v5::properties /*props*/){
            std::cout << "[client] suback received. packet_id: " << packet_id << std::endl;
            for (auto const& e : reasons) {
                switch (e) {
                case MQTT_NS::v5::suback_reason_code::granted_qos_0:
                    std::cout << "[client] subscribe success: qos0" << std::endl;
                    break;
                case MQTT_NS::v5::suback_reason_code::granted_qos_1:
                    std::cout << "[client] subscribe success: qos1" << std::endl;
                    break;
                case MQTT_NS::v5::suback_reason_code::granted_qos_2:
                    std::cout << "[client] subscribe success: qos2" << std::endl;
                    break;
                default:
                    std::cout << "[client] subscribe failed: reason_code = " << static_cast<int>(e) << std::endl;
                    break;
                }
            }
            //if (packet_id == pid_sub1) {
            //    c->publish("mqtt_client_cpp/topic1","test!!!", MQTT_NS::qos::at_most_once);
            //}
            //else if (packet_id == pid_sub2) {
            //    c->publish("mqtt_client_cpp/topic2_1", "test2_1", MQTT_NS::qos::at_least_once);
            //    c->publish("mqtt_client_cpp/topic2_2", "test2_2", MQTT_NS::qos::exactly_once);
            //}
            return true;
        });

    // 发布
    c->set_v5_publish_handler( // use v5 handler
        [&]
        (MQTT_NS::optional<packet_id_t> packet_id,
         MQTT_NS::publish_options pubopts,
         MQTT_NS::buffer topic_name,
         MQTT_NS::buffer contents,
         MQTT_NS::v5::properties /*props*/){
            std::cout << "[client] publish received. "
                      << "dup: "     << pubopts.get_dup()
                      << " qos: "    << pubopts.get_qos()
                      << " retain: " << pubopts.get_retain() << std::endl;
            if (packet_id)
                std::cout << "[client] packet_id: " << *packet_id << std::endl;
            std::cout << "[client] topic_name: " << topic_name << std::endl;
            std::cout << "[client] contents: " << contents << std::endl;
            
            if(topic_name == "/drone/gryo/euler") {
                for(int i = 0;i < 3;i++) {
                    std::istringstream(contents.to_string().substr(i * 5,5)) >> euler[i];
                    std::cout << euler[0] << '\t' << euler[1] << '\t' << euler[2] << '\n';
                }
            }
            
            //disconnect();
            return true;
        });

    // Connect
    c->connect();
    ioc.run();
}