import java.io.IOException;
import java.io.PrintWriter;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServlet;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.junit.Test;

public class MainTest {
    public static class AServlet extends HttpServlet {
        protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
            // 在这里编写处理路径为"/a"的GET请求的逻辑
            PrintWriter out = response.getWriter();
            out.println("Hello from AServlet!");
        }
    }

    public static class AnotherServlet extends HttpServlet {
        protected void doGet(HttpServletRequest request, HttpServletResponse response) throws ServletException, IOException {
            // 在这里编写处理路径为"/b/*"的GET请求的逻辑
            PrintWriter out = response.getWriter();
            out.println("Hello from AnotherServlet! You accessed path: " + request.getPathInfo());
        }
    }
    @Test
    public void testMain() {
        Server server = new Server(8080);

        // 创建 ServletContextHandler
        ServletContextHandler servletContextHandler = new ServletContextHandler();
        servletContextHandler.setContextPath("/");

        // 添加 Servlet
        servletContextHandler.addServlet(new ServletHolder(new AServlet()), "/a");
        servletContextHandler.addServlet(new ServletHolder(new AnotherServlet()), "/b");

        // 将 HandlerList 设置到 Server 中
        server.setHandler(servletContextHandler);

        // 启动 Server
        try {
            server.start();
            server.join();
        } catch(Exception e) {
            e.printStackTrace();;
        }
    }
}