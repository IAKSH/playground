#pragma once
#include <crow.h>
#include "session.hpp"
#include <optional>
#include <cstring>

struct CORS {
    struct context {};
    void before_handle(crow::request& req, crow::response& res, context&);
    void after_handle(crow::request& req, crow::response& res, context&);
};

struct SessionMiddleware {
    struct context {
        std::optional<std::string> session_id;
        SessionState* session = nullptr;
        bool valid = true; // 标记是否有效
    };

    SessionManager* manager = nullptr;

    SessionMiddleware() = default;
    SessionMiddleware(SessionManager* m) : manager(m) {}

    void before_handle(crow::request& req, crow::response& res, context& ctx);
    void after_handle(crow::request& /*req*/, crow::response& /*res*/, context& /*ctx*/);
};
