#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

void init() noexcept {
    auto my_stdout_logger = spdlog::stdout_color_mt("my_stdout_logger");
    my_stdout_logger->info("nihao");

    auto my_stderr_logger = spdlog::stderr_color_mt("my_stderr_logger");    
    my_stderr_logger->info("nihao");

    // 这里使用了spdlog::stdout_color_mt()或spdlog::stderr_color_mt()来创建logger
    // 这会自动将创建的logger注册到全局，所以可以使用spdlog::get()在各处访问
    // 另一种情况见stdout_sink.cpp

    // 另一种创建logger的方法是组合sink创建，见stdout_sink.cpp或file_sink.cpp
}

int main() {
    init();
    spdlog::get("my_stdout_logger")->info("loggers can be retrieved from a global registry using the spdlog::get(logger_name)");
    spdlog::get("my_stderr_logger")->info("loggers can be retrieved from a global registry using the spdlog::get(logger_name)");
    return 0;
}
