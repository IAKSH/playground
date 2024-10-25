const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/rotate',
    createProxyMiddleware({
      target: 'http://192.168.189.93:18080',
      changeOrigin: true,
    })
  );
};
