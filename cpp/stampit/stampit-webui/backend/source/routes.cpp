#include <routes.hpp>

std::optional<route::MultipartFile> route::parse_multipart(const std::string& body, const std::string& content_type) {
    auto pos = content_type.find("boundary=");
    if (pos == std::string::npos) return std::nullopt;
    std::string boundary = "--" + content_type.substr(pos + 9);

    std::vector<std::string> parts;
    size_t start = 0;
    while (true) {
        size_t bpos = body.find(boundary, start);
        if (bpos == std::string::npos) break;
        size_t bend = body.find("\r\n", bpos);
        if (bend == std::string::npos) break;
        size_t next = bend + 2;
        size_t bnext = body.find(boundary, next);
        if (bnext == std::string::npos) break;
        parts.emplace_back(body.substr(next, bnext - next));
        start = bnext;
    }

    for (auto& part : parts) {
        size_t header_end = part.find("\r\n\r\n");
        if (header_end == std::string::npos) continue;
        std::string headers = part.substr(0, header_end);
        std::string payload = part.substr(header_end + 4);

        if (!payload.empty() && payload.size() >= 2 &&
            payload.substr(payload.size() - 2) == "\r\n") {
            payload.resize(payload.size() - 2);
        }

        auto cd_pos = headers.find("Content-Disposition:");
        if (cd_pos == std::string::npos) continue;
        auto fn_pos = headers.find("filename=");
        if (fn_pos == std::string::npos) continue;

        std::string filename;
        size_t start_q = headers.find('"', fn_pos);
        if (start_q != std::string::npos) {
            size_t end_q = headers.find('"', start_q + 1);
            if (end_q != std::string::npos) {
                filename = headers.substr(start_q + 1, end_q - (start_q + 1));
            }
        } else {
            size_t start_u = fn_pos + 9;
            size_t end_u = headers.find(';', start_u);
            if (end_u == std::string::npos) end_u = headers.find('\r', start_u);
            if (end_u == std::string::npos) end_u = headers.size();
            filename = headers.substr(start_u, end_u - start_u);
        }

        MultipartFile mf;
        mf.filename = filename;
        mf.data.assign(payload.begin(), payload.end());
        return mf;
    }

    return std::nullopt;
}

bool route::write_file(const std::string& path, const std::vector<uint8_t>& buf) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) return false;
    ofs.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return ofs.good();
}

bool route::save_full_png(stampit::Image& img, const std::string& path) {
    if (img.empty()) return false;
    std::string out = path;
    if (fs::path(out).extension().empty()) out += ".png";
    img.save(out);
    std::error_code ec;
    return fs::exists(out, ec) && fs::file_size(out, ec) > 0;
}

// session_id extractors (query or header)
std::optional<std::string> route::extract_session_id_from_query_or_header(const crow::request& req) {
    auto q = req.url_params.get("session_id");
    if (q && std::strlen(q) > 0) return std::string(q);
    try {
        auto h = req.get_header_value("X-Session-Id");
        if (!h.empty()) return h;
    } catch (...) {}
    return std::nullopt;
}

std::optional<std::string> route::extract_session_id_from_json(const crow::json::rvalue& body) {
    if (!body) return std::nullopt;
    if (!body.has("session_id")) return std::nullopt;
    return std::optional<std::string>(body["session_id"].s());
}

// helper: update last_access timestamp
void route::touch_session(SessionState* state) {
    if (!state) return;
    state->last_access = std::chrono::system_clock::now();
}
