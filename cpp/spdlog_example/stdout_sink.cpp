#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/stdout_sinks.h>

void init_logger() noexcept {
    // sink是logger的输出对象，需要配合logger使用
    auto console_sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("my_logger", console_sink);
    // 此处创建的logger是通过std::make_shared<spdlog::logger>()创建的，std::make_shared自然不会将其注册到全局
    // 需要使用spdlog::register_logger()进行手动注册，才能使用spdlog::get()在各处访问
    spdlog::register_logger(logger);

    logger->set_level(spdlog::level::info);
 
    logger->info("Hello, ({}) spdlog!","no color");
}

void init_colored_logger() noexcept {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto logger = std::make_shared<spdlog::logger>("my_colored_logger", console_sink);
    spdlog::register_logger(logger);

    logger->set_level(spdlog::level::info);
 
    logger->info("Hello, spdlog!");
}

int main() {
    init_logger();
    init_colored_logger();

    spdlog::get("my_logger")->info("loggers can be retrieved from a global registry using the spdlog::get(logger_name)");
    spdlog::get("my_colored_logger")->info("loggers can be retrieved from a global registry using the spdlog::get(logger_name)");

    return 0;
}
