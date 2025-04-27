#include <crow_all.h>

struct LocalAdminAreaGuard : crow::ILocalMiddleware {
    struct context {};

    void before_handle(crow::request& req, crow::response& res, context& ctx) {
        if (req.remote_ip_address != "192.168.1.109") {
            CROW_LOG_WARNING << "kicked connection from " << req.remote_ip_address;
            res.code = 403;
            res.end();
        }
    }

    void after_handle(crow::request& req, crow::response& res, context& ctx) {
        //CROW_LOG_INFO << "middleware after_handle: " << req.body;
    }
};

int main() {
    // 创建一个Crow应用
    crow::App<LocalAdminAreaGuard> app;

    // 创建一个Blueprint
    crow::Blueprint bp("bp_prefix", "./custom_static", "./custom_templates");

    // 在Blueprint中定义一个路由
    CROW_BP_ROUTE(bp, "/xyz")([](){
        return "Hello, world!";
    });

    CROW_BP_ROUTE(bp, "/html/<string>")([](std::string name) {
        auto page = crow::mustache::load("with_arg.html");
        crow::mustache::context ctx ({{"person", name}});
        return page.render(ctx);
    });

    CROW_BP_ROUTE(bp, "/sayori")([](){
        crow::response res;
        //res.set_static_file_info("Sayori_Sticker_Excited.webp");//不可用
        res.set_static_file_info("./custom_static/Sayori_Sticker_Excited.webp");//可用
        return res;
    });

    // 注册Blueprint到app，以及向其注入中间件
    bp.CROW_MIDDLEWARES(app,LocalAdminAreaGuard);
    app.register_blueprint(bp);

    // 运行应用
    app.port(18080).multithreaded().run();

    return 0;
}