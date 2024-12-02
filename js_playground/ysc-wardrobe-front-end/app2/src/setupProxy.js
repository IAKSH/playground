const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function (app) {
    app.use(
        '/api',
        createProxyMiddleware({
            target: 'http://www.yunsc.asia', // 代理的目标地址
            changeOrigin: true,
            pathRewrite: {
                '^/api': ''  // 将请求路径中的 "/api" 替换为空
            }
        })
    );
};
