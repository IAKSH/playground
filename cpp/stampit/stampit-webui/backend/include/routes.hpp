#pragma once

#include <crow.h>
#include "middleware.hpp"
#include "session.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <chrono>

namespace route {
    namespace fs = std::filesystem;

    // Minimal multipart parser (single-file field)
    struct MultipartFile {
        std::string filename;
        std::vector<uint8_t> data;
    };

    std::optional<MultipartFile> parse_multipart(const std::string& body, const std::string& content_type);
    bool write_file(const std::string& path, const std::vector<uint8_t>& buf);
    bool save_full_png(stampit::Image& img, const std::string& path);
    std::optional<std::string> extract_session_id_from_query_or_header(const crow::request& req);
    std::optional<std::string> extract_session_id_from_json(const crow::json::rvalue& body);
    void touch_session(SessionState* state);

    template<typename App>
    inline void regist(App& app, SessionManager& manager) {
        // POST /image/upload
        CROW_ROUTE(app, "/image/upload").methods(crow::HTTPMethod::POST)
        ([&manager](const crow::request& req) {
            auto sid_opt = extract_session_id_from_query_or_header(req);
            std::string session_id;
            SessionState* state = nullptr;

            if (sid_opt) {
                session_id = *sid_opt;
                state = manager.get_or_create_session(session_id);
            } else {
                session_id = manager.create_session();
                state = manager.get_session(session_id);
            }
            if (!state) return crow::response(500, "Failed to create session");

            auto ct = req.get_header_value("Content-Type");
            if (ct.empty() || ct.find("multipart/form-data") == std::string::npos) {
                return crow::response(400, "Expected multipart/form-data");
            }

            auto mf = parse_multipart(req.body, ct);
            if (!mf) return crow::response(400, "Invalid multipart or missing file");

            fs::create_directories(state->base_dir + "/uploads");
            std::string temp_path = state->base_dir + "/uploads/" + (mf->filename.empty() ? std::string("upload.bin") : mf->filename);
            if (!write_file(temp_path, mf->data)) {
                return crow::response(500, "Failed to write upload");
            }

            {
                std::lock_guard<std::mutex> lock(state->mtx);
                stampit::Image empty;
                state->original_img = empty;
                state->thresholded_img = empty;
                state->final_img = empty;

                state->original_img.load(temp_path);
                if (state->original_img.empty()) {
                    return crow::response(400, "Failed to load image");
                }

                state->auto_rect.reset();
                state->has_processed = false;
                touch_session(state);
            }

            crow::json::wvalue res;
            res["status"] = "ok";
            res["width"] = state->original_img.width();
            res["height"] = state->original_img.height();
            res["filename"] = mf->filename;
            res["session_id"] = session_id;
            return crow::response{res};
        });

        // POST /image/process
        CROW_ROUTE(app, "/image/process").methods(crow::HTTPMethod::POST)
        ([&manager](const crow::request& req) {
            auto body = crow::json::load(req.body);
            if (!body) return crow::response(400, "Invalid JSON");

            auto sid_opt = extract_session_id_from_json(body);
            if (!sid_opt) return crow::response(400, "Missing session_id in JSON");
            std::string session_id = *sid_opt;

            SessionState* state = manager.get_session(session_id);
            if (!state) return crow::response(404, "Invalid session");

            int threshold = body.has("threshold") ? body["threshold"].i() : 0;
            int min_area = body.has("min_area") ? body["min_area"].i() : 0;
            int margin = body.has("margin") ? body["margin"].i() : 10;

            {
                std::lock_guard<std::mutex> lock(state->mtx);
                if (state->original_img.empty()) return crow::response(400, "No image uploaded");

                stampit::Image img = state->original_img.copy();
                img.threshold(threshold);
                {
                    stampit::Image th = img.copy();
                    state->thresholded_img = th;
                }
                img.denoise(min_area);
                auto rect = img.get_min_boundary(margin);
                state->auto_rect = rect;
                img.set_color(255, 0, 0);
                {
                    stampit::Image fin = img.copy();
                    state->final_img = fin;
                }

                state->last_threshold = threshold;
                state->last_min_area = min_area;
                state->has_processed = true;
                touch_session(state);

                crow::json::wvalue res;
                res["status"] = "ok";
                res["threshold"] = threshold;
                res["min_area"] = min_area;
                res["margin"] = margin;
                res["boundary"] = {
                    {"x", rect.x},
                    {"y", rect.y},
                    {"w", rect.w},
                    {"h", rect.h}
                };
                res["final"] = {
                    {"width", state->final_img.width()},
                    {"height", state->final_img.height()}
                };
                res["session_id"] = session_id;
                return crow::response{res};
            }
        });

        // GET /image/thresholded?session_id=...
        CROW_ROUTE(app, "/image/thresholded").methods(crow::HTTPMethod::GET)
        ([&manager](const crow::request& req) {
            auto sid_opt = extract_session_id_from_query_or_header(req);
            if (!sid_opt) return crow::response(400, "Missing session_id");
            SessionState* state = manager.get_session(*sid_opt);
            if (!state) return crow::response(404, "Invalid session");

            {
                std::lock_guard<std::mutex> lock(state->mtx);
                if (state->thresholded_img.empty()) return crow::response(404, "No thresholded image");
                touch_session(state);

                fs::create_directories(state->base_dir + "/cache");
                std::string out_path = state->base_dir + "/cache/thresholded.png";
                stampit::Image tmp = state->thresholded_img.copy();
                if (!save_full_png(tmp, out_path)) {
                    return crow::response(500, "Failed to encode thresholded image");
                }

                std::ifstream ifs(out_path, std::ios::binary);
                std::ostringstream oss;
                oss << ifs.rdbuf();
                auto resp = crow::response(oss.str());
                resp.set_header("Content-Type", "image/png");
                return resp;
            }
        });

        // GET /image/final?session_id=...
        CROW_ROUTE(app, "/image/final").methods(crow::HTTPMethod::GET)
        ([&manager](const crow::request& req) {
            auto sid_opt = extract_session_id_from_query_or_header(req);
            if (!sid_opt) return crow::response(400, "Missing session_id");
            SessionState* state = manager.get_session(*sid_opt);
            if (!state) return crow::response(404, "Invalid session");

            {
                std::lock_guard<std::mutex> lock(state->mtx);
                if (state->final_img.empty()) return crow::response(404, "No final image");
                touch_session(state);

                fs::create_directories(state->base_dir + "/cache");
                std::string out_path = state->base_dir + "/cache/final.png";
                stampit::Image tmp = state->final_img.copy();
                if (!save_full_png(tmp, out_path)) {
                    return crow::response(500, "Failed to encode final image");
                }

                std::ifstream ifs(out_path, std::ios::binary);
                std::ostringstream oss;
                oss << ifs.rdbuf();
                auto resp = crow::response(oss.str());
                resp.set_header("Content-Type", "image/png");
                return resp;
            }
        });

        // GET /image/boundary?session_id=...
        CROW_ROUTE(app, "/image/boundary").methods(crow::HTTPMethod::GET)
        ([&manager](const crow::request& req) {
            auto sid_opt = extract_session_id_from_query_or_header(req);
            if (!sid_opt) return crow::response(400, "Missing session_id");
            SessionState* state = manager.get_session(*sid_opt);
            if (!state) return crow::response(404, "Invalid session");

            {
                std::lock_guard<std::mutex> lock(state->mtx);
                if (!state->auto_rect.has_value()) return crow::response(404, "No boundary");
                touch_session(state);

                auto rect = state->auto_rect.value();
                crow::json::wvalue res;
                res["x"] = rect.x;
                res["y"] = rect.y;
                res["w"] = rect.w;
                res["h"] = rect.h;
                res["session_id"] = *sid_opt;
                return crow::response{res};
            }
        });

        // POST /image/save
        CROW_ROUTE(app, "/image/save").methods(crow::HTTPMethod::POST)
        ([&manager](const crow::request& req) {
            auto body = crow::json::load(req.body);
            if (!body) return crow::response(400, "Invalid JSON");

            auto sid_opt = extract_session_id_from_json(body);
            if (!sid_opt) return crow::response(400, "Missing session_id in JSON");
            std::string session_id = *sid_opt;

            SessionState* state = manager.get_session(session_id);
            if (!state) return crow::response(404, "Invalid session");

            int x = body.has("x") ? body["x"].i() : 0;
            int y = body.has("y") ? body["y"].i() : 0;
            int w = body.has("w") ? body["w"].i() : 0;
            int h = body.has("h") ? body["h"].i() : 0;

            {
                std::lock_guard<std::mutex> lock(state->mtx);
                if (state->final_img.empty()) return crow::response(400, "No final image");
                touch_session(state);

                fs::create_directories(state->base_dir + "/cache");
                std::string tmp = state->base_dir + "/cache/crop.png";
                state->final_img.save_with_transparent_pix(tmp, x, y, w, h);

                std::ifstream ifs(tmp, std::ios::binary);
                std::ostringstream oss;
                oss << ifs.rdbuf();
                crow::response resp(oss.str());
                resp.set_header("Content-Type", "image/png");
                resp.set_header("Content-Disposition", "attachment; filename=\"crop.png\"");
                return resp;
            }
        });

        // GET /image/status?session_id=...
        CROW_ROUTE(app, "/image/status").methods(crow::HTTPMethod::GET)
        ([&manager](const crow::request& req) {
            auto sid_opt = extract_session_id_from_query_or_header(req);
            if (!sid_opt) return crow::response(400, "Missing session_id");
            SessionState* state = manager.get_session(*sid_opt);
            if (!state) return crow::response(404, "Invalid session");

            {
                std::lock_guard<std::mutex> lock(state->mtx);
                touch_session(state);
                crow::json::wvalue res;
                res["has_original"] = !state->original_img.empty();
                res["has_thresholded"] = !state->thresholded_img.empty();
                res["has_final"] = !state->final_img.empty();
                res["has_processed"] = state->has_processed;
                res["last_threshold"] = state->last_threshold;
                res["last_min_area"] = state->last_min_area;
                if (state->auto_rect.has_value()) {
                    auto r = state->auto_rect.value();
                    res["boundary"] = { {"x", r.x}, {"y", r.y}, {"w", r.w}, {"h", r.h} };
                } else {
                    res["boundary"] = nullptr;
                }
                res["session_id"] = *sid_opt;
                return crow::response{res};
            }
        });
    }
}