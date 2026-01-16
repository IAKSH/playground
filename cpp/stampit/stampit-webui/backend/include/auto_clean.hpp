#pragma once
#include "session.hpp"
#include <thread>
#include <atomic>
#include <condition_variable>
#include <filesystem>
#include <spdlog/spdlog.h>

namespace fs = std::filesystem;

class SessionCleaner {
public:
    SessionCleaner(SessionManager& mgr,
                   std::chrono::seconds ttl,
                   std::chrono::seconds interval,
                   const std::string& staticDir,
                   std::shared_ptr<spdlog::logger> logger)
        : manager(mgr), TTL(ttl), INTERVAL(interval),
          static_dir(staticDir), stopFlag(false), logger(std::move(logger)) {}

    void start();
    void stop();

private:
    SessionManager& manager;
    std::chrono::seconds TTL;
    std::chrono::seconds INTERVAL;
    std::string static_dir;
    std::shared_ptr<spdlog::logger> logger;

    std::atomic<bool> stopFlag;
    std::mutex mtx;
    std::condition_variable cv;
    std::thread worker;

    void run();
    void clear_cache();
};
