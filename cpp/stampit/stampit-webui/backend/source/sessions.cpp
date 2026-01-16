#include <session.hpp>

SessionManager::SessionManager() {
    fs::create_directories("sessions");
}

std::string SessionManager::create_session() {
    std::lock_guard<std::mutex> lock(mtx_);
    std::string id = generateId();
    sessions_.try_emplace(id); // default-construct SessionState in-place
    sessions_[id].base_dir = "sessions/" + id;
    fs::create_directories(sessions_[id].base_dir);
    return id;
}

SessionState* SessionManager::get_or_create_session(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(id);
    if (it != sessions_.end()) return &it->second;
    sessions_.try_emplace(id);
    sessions_[id].base_dir = "sessions/" + id;
    fs::create_directories(sessions_[id].base_dir);
    return &sessions_[id];
}

SessionState* SessionManager::get_session(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(id);
    if (it == sessions_.end()) return nullptr;
    // update last_access here to reflect retrieval
    it->second.last_access = std::chrono::system_clock::now();
    return &it->second;
}

bool SessionManager::remove_session(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = sessions_.find(id);
    if (it == sessions_.end()) return false;
    std::error_code ec;
    fs::remove_all(it->second.base_dir, ec);
    sessions_.erase(it);
    return true;
}

int SessionManager::cleanup_expired(std::chrono::seconds ttl) {
    std::vector<std::string> to_remove;
    auto now = std::chrono::system_clock::now();
    {
        std::lock_guard<std::mutex> lock(mtx_);
        for (auto& kv : sessions_) {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(now - kv.second.last_access);
            if (age > ttl) to_remove.push_back(kv.first);
        }
    }
    int removed = 0;
    for (auto& id : to_remove) {
        if (remove_session(id)) ++removed;
    }
    return removed;
}

std::string SessionManager::generateId() {
    static std::mt19937 rng{std::random_device{}()};
    static std::uniform_int_distribution<int> dist(0, 15);
    std::string id;
    id.reserve(32);
    for (int i = 0; i < 32; ++i) id += "0123456789abcdef"[dist(rng)];
    return id;
}
