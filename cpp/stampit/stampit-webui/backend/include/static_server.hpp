#pragma once

#include <crow.h>
#include <filesystem>
#include <string_view>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <mutex>

namespace static_route {
    namespace fs = std::filesystem;

    inline std::string content_type_for(const std::string& path) {
        static const std::unordered_map<std::string, std::string> map = {
            {".html", "text/html; charset=utf-8"},
            {".htm",  "text/html; charset=utf-8"},
            {".js",   "application/javascript; charset=utf-8"},
            {".css",  "text/css; charset=utf-8"},
            {".png",  "image/png"},
            {".jpg",  "image/jpeg"},
            {".jpeg", "image/jpeg"},
            {".gif",  "image/gif"},
            {".svg",  "image/svg+xml"},
            {".json", "application/json; charset=utf-8"},
            {".wasm", "application/wasm"},
            {".ico",  "image/x-icon"}
        };
        auto ext = fs::path(path).extension().string();
        auto it = map.find(ext);
        if (it != map.end()) return it->second;
        return "application/octet-stream";
    }

    // 简单缓存：文件路径 -> 文件内容
    struct StaticCache {
        std::unordered_map<std::string, std::string> cache;
        std::mutex mtx;

        bool get(const std::string& path, std::string& out) {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = cache.find(path);
            if (it != cache.end()) {
                out = it->second;
                return true;
            }
            return false;
        }

        void put(const std::string& path, const std::string& data) {
            std::lock_guard<std::mutex> lock(mtx);
            cache[path] = data;
        }
    };

    template<typename App>
    inline void regist(App& app, std::string_view static_root) {
        static StaticCache cache;

        auto read_file = [&](const fs::path& p, std::string& out) -> bool {
            if (cache.get(p.string(), out)) return true;
            std::error_code ec;
            if (!fs::exists(p, ec) || !fs::is_regular_file(p, ec)) return false;
            std::ifstream ifs(p, std::ios::binary);
            if (!ifs) return false;
            std::ostringstream oss;
            oss << ifs.rdbuf();
            out = oss.str();
            cache.put(p.string(), out);
            return true;
        };

        // 根路径和 /index.html
        CROW_ROUTE(app, "/")
        ([static_root, read_file]() {
            fs::path index = fs::path(static_root) / "index.html";
            std::string body;
            if (!read_file(index, body)) {
                return crow::response(500, "index.html not found");
            }
            crow::response r(body);
            r.set_header("Content-Type", "text/html; charset=utf-8");
            r.set_header("Cache-Control", "no-cache");
            return r;
        });

        CROW_ROUTE(app, "/index.html")
        ([static_root, read_file]() {
            fs::path index = fs::path(static_root) / "index.html";
            std::string body;
            if (!read_file(index, body)) {
                return crow::response(500, "index.html not found");
            }
            crow::response r(body);
            r.set_header("Content-Type", "text/html; charset=utf-8");
            r.set_header("Cache-Control", "no-cache");
            return r;
        });

        // 通用静态文件路由
        CROW_ROUTE(app, "/<path>")
        ([static_root, read_file](const crow::request& req, std::string path) {
            fs::path requested = fs::path(static_root) / path;
            std::string body;
            if (read_file(requested, body)) {
                crow::response r(body);
                r.set_header("Content-Type", content_type_for(requested.string()));
                r.set_header("Cache-Control", "public, max-age=31536000, immutable");
                return r;
            }
            // fallback 到 index.html (SPA)
            fs::path index = fs::path(static_root) / "index.html";
            if (!read_file(index, body)) {
                return crow::response(500, "index.html not found");
            }
            crow::response r(body);
            r.set_header("Content-Type", "text/html; charset=utf-8");
            r.set_header("Cache-Control", "no-cache");
            return r;
        });
    }
}
