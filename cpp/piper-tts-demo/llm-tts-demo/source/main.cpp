#include <iostream>
#include <thread>
#include <spdlog/spdlog.h>
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <tone/player.hpp>
#include <tone/synth.hpp>
#include <tone/normalizer.hpp>
#include <tone/splitter.hpp>

static constexpr std::string_view MODEL {"../../model/piper/zh_CN-huayan-medium.onnx"};
static constexpr std::string_view ESPEAK {"./vendor/espeak-ng/espeak-ng-data"};
static constexpr std::string_view LLM_SERVER{"http://localhost:80"};
static constexpr std::string_view LLM_API{"/llm/local/chat"};
//static constexpr std::string_view LLM_API{"/llm/deepseek/chat"};

/*
{
    "version": "v1",
    "session": "xxxxxx",
    "message": "xxxxx",
    "token_limit": 128
}
*/

int main() {
    spdlog::set_level(spdlog::level::debug);

    std::string session = "none";
    std::string message;

    httplib::Client cli(LLM_SERVER.data());
    nlohmann::json req_json = {
        {"version","v1"},
        {"session","none"},
        {"message", "你好，介绍你自己。"},
        {"token_limit", 1024}
    };

    TextNormalizer normalizer;
    SentenceSplitter splitter;
    Synthesizer synth(MODEL,ESPEAK);
    ALSAPlayer player;
    player.set_speed(1.5f);

    while(true) {
        std::getline(std::cin,message);
        req_json["message"] = message;
        spdlog::debug(req_json.dump());
        auto res = cli.Post(LLM_API.data(),req_json.dump(),"application/json");
        if(res) {
            spdlog::debug(res->body);

            auto json_res = nlohmann::json::parse(res->body);
            session = json_res["session"].get<std::string>();
            message = json_res["reply"].get<std::string>();
            
            auto normalized = normalizer.normalize(message);
            auto sentences = splitter.split(normalized);

            req_json["session"] = session;

            spdlog::info("session: {}", session);
            spdlog::info("replay:\n{}", message);

            for(const auto& s : sentences) {
                player.prepare();
                synth.generate(s,[&player](const float *pcm,size_t pcm_len){
                    player.play(pcm,pcm_len);
                });
                player.drain();
                while(!player.finished()) {
                    spdlog::debug("unfinished!");
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        }
        else {
            spdlog::error("Request failed");
        }
    }

    return 0;
}
