#pragma once
#include "session.hpp"
#include <unordered_map>
#include <mutex>
#include <random>
#include <filesystem>
#include <chrono>
#include <optional>
#include <string>
#include <stampit/image.hpp>

namespace fs = std::filesystem;

struct SessionState {
    std::mutex mtx;
    stampit::Image original_img;
    stampit::Image thresholded_img;
    stampit::Image final_img;
    std::optional<stampit::Rect> auto_rect;

    int last_threshold = 0;
    int last_min_area = 0;
    bool has_processed = false;

    std::string base_dir; // sessions/<session_id>
    std::chrono::system_clock::time_point last_access = std::chrono::system_clock::now();
};

class SessionManager {
public:
    SessionManager();
    // Create a new random session id and SessionState
    std::string create_session();
    SessionState* get_or_create_session(const std::string& id);
    // Get existing session, or nullptr
    SessionState* get_session(const std::string& id);
    // Remove session from map and delete disk directory
    bool remove_session(const std::string& id);
    // Cleanup expired sessions older than ttl; returns number removed
    int cleanup_expired(std::chrono::seconds ttl);

private:
    std::unordered_map<std::string, SessionState> sessions_;
    std::mutex mtx_;
    std::string generateId();
};
