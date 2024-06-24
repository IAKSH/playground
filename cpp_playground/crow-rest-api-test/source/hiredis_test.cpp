#include <iostream>
#include <hiredis/hiredis.h>

int main() {
    // 创建一个redisContext对象，连接到Redis服务器
    redisContext* c = redisConnect("127.0.0.1", 6379);
    if (c == NULL || c->err) {
        if (c) {
            std::cout << "Error: " << c->errstr << std::endl;
            // handle error
        } else {
            std::cout << "Can't allocate redis context" << std::endl;
        }
        return 1;
    }

    // 执行SET命令
    redisReply* reply = (redisReply*)redisCommand(c, "SET %s %s", "key", "value");
    if (reply == NULL) {
        std::cout << "Error: " << c->errstr << std::endl;
        redisFree(c);
        return 1;
    }
    freeReplyObject(reply);

    // 执行GET命令
    reply = (redisReply*)redisCommand(c, "GET %s", "key");
    if (reply == NULL) {
        std::cout << "Error: " << c->errstr << std::endl;
        redisFree(c);
        return 1;
    }
    std::cout << "GET key: " << reply->str << std::endl;
    freeReplyObject(reply);

    // 释放连接
    redisFree(c);

    return 0;
}
