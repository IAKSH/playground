#include <middleware.hpp>

void CORS::before_handle(crow::request& req, crow::response& res, context&) {

}

void CORS::after_handle(crow::request& req, crow::response& res, context&) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, X-Session-Id");
}

void SessionMiddleware::before_handle(crow::request& req, crow::response& res, context& ctx) {
    // 提取 session_id
    auto q = req.url_params.get("session_id");
    if (q && std::strlen(q) > 0) {
        ctx.session_id = std::string(q);
    } else {
        try {
            auto h = req.get_header_value("X-Session-Id");
            if (!h.empty()) ctx.session_id = h;
        } catch (...) {}
    }

    if (ctx.session_id) {
        ctx.session = manager->get_session(*ctx.session_id);
        if (!ctx.session) {
            res.code = 404;
            res.write("Invalid session");
            ctx.valid = false;
            res.end();
            return;
        }
        // 更新访问时间
        std::lock_guard<std::mutex> lock(ctx.session->mtx);
        ctx.session->last_access = std::chrono::system_clock::now();
    }
}

void SessionMiddleware::after_handle(crow::request& /*req*/, crow::response& /*res*/, context& /*ctx*/) {}
