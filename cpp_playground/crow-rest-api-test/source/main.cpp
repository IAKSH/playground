#define CROW_ENFORCE_WS_SPEC
#define CROW_ENABLE_SSL
#define CROW_ENABLE_COMPRESSION

#include <crow_all.h>

int main() {
    crow::SimpleApp app;
    // set log level
    //app.loglevel(crow::LogLevel::Warning);

    // basical
    CROW_ROUTE(app, "/")([](){
        return "Hello world\nThis hardware have " + std::to_string(std::thread::hardware_concurrency()) + " threads";
    });

    // static html
    CROW_ROUTE(app, "/html")([](){
        auto page = crow::mustache::load_text("index.html");
        return page;
    });

    CROW_ROUTE(app, "/html/ws")([](){
        auto page = crow::mustache::load_text("test_ws.html");
        return page;
    });

    CROW_ROUTE(app, "/html/json")([](){
        auto page = crow::mustache::load_text("test_json.html");
        return page;
    });

    //static html with args
    CROW_ROUTE(app, "/html/<string>")([](std::string name) {
        auto page = crow::mustache::load("with_arg.html");
        crow::mustache::context ctx ({{"person", name}});
        return page.render(ctx);
    });

    // json
    CROW_ROUTE(app, "/json")
    ([]{
        return crow::json::wvalue({{"from","admin"},{"to","you"},{"message","hello!"}});
    });

    // with arg
    CROW_ROUTE(app,"/hello/<int>")
    ([](int count){
        if (count > 100)
            return crow::response(crow::status::BAD_REQUEST);
        std::ostringstream os;
        os << count << " bottles of beer!";
        return crow::response(os.str());
    });

    // with more args
    CROW_ROUTE(app, "/add/<int>/<int>")
    ([](int a, int b)
    {
        return std::to_string(a+b);
    });

    // with wrong num of arg
    //// Compile error with message "Handler type is mismatched with URL parameters"
    //CROW_ROUTE(app,"/another/<int>")
    //([](int a, int b){
    //    return crow::response(500);
    //});

    // HTTP POST
    CROW_ROUTE(app, "/add_json")
    //.methods("POST"_method)
    .methods(crow::HTTPMethod::POST)
    ([](const crow::request& req){
        auto x = crow::json::load(req.body);
        if (!x)
            return crow::response(crow::status::BAD_REQUEST); // same as crow::response(400)
        int sum = x["a"].i()+x["b"].i();
        std::ostringstream os;
        os << sum;
        return crow::response{os.str()};
    });

    // static resource
    CROW_ROUTE(app, "/sayori")([](){
        crow::response res;
        res.set_static_file_info("static/Sayori_Sticker_Excited.webp");
        return res;
    });

    // Query String
    CROW_ROUTE(app, "/params")([](const crow::request& req){
        std::string param1 = req.url_params.get("param1");
        std::string param2 = req.url_params.get("param2");
        return crow::response(200, "Param1: " + param1 + ", Param2: " + param2);
    });

    CROW_ROUTE(app, "/params_list")([](const crow::request& req){
        std::string response = "All parameters: ";
        auto key_values = req.url_params.get_list("key");
        response += "\nlength of key_values: ";
        response += '0' + key_values.size();
        response += "\nkey's values: ";
        for (const auto& value : key_values) {
            response += value;
            response += ", ";
        }
        return crow::response(200, response);
    });

    CROW_ROUTE(app, "/params_dict")([](const crow::request& req){
        std::string response = "All parameters: ";
        auto key_values = req.url_params.get_dict("key");
        response += "\nkey.sub_key1 = ";
        response += key_values["sub_key1"];
        response += "\nkey.sub_key2 = ";
        response += key_values["sub_key2"];
        auto another_key_values = req.url_params.get_dict("another_key");
        response += "\nanother_key.sub_key1 = ";
        response += another_key_values["sub_key1"];
        return crow::response(200, response);
    });

    CROW_WEBSOCKET_ROUTE(app, "/ws")
    .onopen([&](crow::websocket::connection& conn){
            conn.send_text("hello");
        })
    .onclose([&](crow::websocket::connection& conn, const std::string& reason){
            conn.send_text("bye");
        })
    .onmessage([&](crow::websocket::connection& conn, const std::string& data, bool is_binary){
            conn.send_text(is_binary ? "recieved binary" : "recieved str");
            conn.send_text(data);
            CROW_LOG_INFO << "ws: " << data;
        });

    CROW_CATCHALL_ROUTE(app)([](){
        return "no page for you!";
    });

    app.port(18080)
        .ssl_file("./tls/server.crt","./tls/server.key")
        .use_compression(crow::compression::algorithm::GZIP)
        .multithreaded()
        .run();
}