#include <iostream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

/**
 * 参考：
 * https://github.com/gabime/spdlog/blob/v1.x/README.md
 * https://www.cnblogs.com/jinyunshaobing/p/16797330.html
*/

void init_logger() noexcept {
    // sink是logger的输出对象，需要配合logger使用
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("test-log.txt");
    // 每天14:22在logs/下创建新的文件
    //auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("test-log.txt","logs/",14,22);
    // 轮转文件，单个写满自动切换到下一个
    // 第二个参数是单文件大小上限，第三个参数是文件数量最大值
    //auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("test-log.txt",1024 * 1024 * 10, 100, false);

    auto logger = std::make_shared<spdlog::logger>("my_logger", file_sink);
    // 此处创建的logger是通过std::make_shared<spdlog::logger>()创建的，std::make_shared自然不会将其注册到全局
    // 需要使用spdlog::register_logger()进行手动注册，才能使用spdlog::get()在各处访问
    spdlog::register_logger(logger);

    logger->set_level(spdlog::level::info);
 
    logger->info("Hello, ({}) spdlog!","no color");
}

int main() {
    init_logger();
    spdlog::get("my_logger")->info("loggers can be retrieved from a global registry using the spdlog::get(logger_name)");
    return 0;
}