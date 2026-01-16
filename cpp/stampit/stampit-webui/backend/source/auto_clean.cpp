#include <auto_clean.hpp>

void SessionCleaner::start() {
    worker = std::thread([this]() { run(); });
    logger->info("running");
}

void SessionCleaner::stop() {
    {
            std::lock_guard<std::mutex> lock(mtx);
            stopFlag.store(true);
    }
    cv.notify_all();
    if (worker.joinable()) worker.join();
    clear_cache();
}

void SessionCleaner::run() {
    std::unique_lock<std::mutex> lock(mtx);
    while (!stopFlag.load()) {
        if (cv.wait_for(lock, INTERVAL, [this]{ return stopFlag.load(); })) {
            break;
        }
        try {
            int removed = manager.cleanup_expired(TTL);
            if (removed > 0) {
                logger->info("removed {} expired sessions", removed);
            }
        } catch (const std::exception& e) {
            logger->error("exception: {}", e.what());
        }
    }
}

void SessionCleaner::clear_cache() {
    try {
        fs::path cacheDir("sessions");
        logger->info("clearing session cache in {}",cacheDir.string());
        if (fs::exists(cacheDir)) {
            fs::remove_all(cacheDir);
            logger->info("cleared");
        }
        else {
            logger->info("nothing to do");
        }
    } catch (const std::exception& e) {
        logger->error("failed to clear session cache: {}", e.what());
    }
}
