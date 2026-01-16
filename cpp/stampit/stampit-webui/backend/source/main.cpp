#include <crow.h>
#include "session.hpp"
#include "middleware.hpp"
#include "routes.hpp"
#include "static_server.hpp"
#include "auto_clean.hpp"
#include <iostream>
#include <csignal>
#include <spdlog/sinks/stdout_color_sinks.h>

#ifndef STATIC_DIR
#define STATIC_DIR "static"
#endif

inline std::string get_static_dir() {
    if (const char* env = std::getenv("STATIC_DIR"))
        return std::string(env);
    return std::string(STATIC_DIR);
}

int main(int argc, char* argv[]) {
    std::string static_dir = get_static_dir();
    spdlog::info("using static files from: {}", static_dir);

    SessionManager manager;
    crow::App<CORS, SessionMiddleware> app;
    app.get_middleware<SessionMiddleware>().manager = &manager;

    route::regist(app, manager);
    static_route::regist(app, static_dir);

    // 启动后台清理器
    SessionCleaner cleaner(manager,
        std::chrono::hours(24),
        std::chrono::minutes(10),
        static_dir,
        spdlog::stdout_color_mt("session-cleaner")
    );
    cleaner.start();

    uint16_t port = 18080;
    if (argc > 1) {
        try { port = static_cast<uint16_t>(std::stoi(argv[1])); }
        catch (...) { spdlog::warn("Invalid port argument, using default 18080"); }
    }

    // 捕获 Ctrl-C / kill
    //std::signal(SIGINT, [](int){ spdlog::info("Received SIGINT, stopping server..."); });
    //std::signal(SIGTERM, [](int){ spdlog::info("Received SIGTERM, stopping server..."); });

    app.port(port).multithreaded().run();

    // Crow run() 返回后，停止清理器
    cleaner.stop();

    return 0;
}
