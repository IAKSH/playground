package me.iaksh;

import me.iaksh.controller.*;
import org.eclipse.jetty.server.Server;
import org.eclipse.jetty.servlet.ServletContextHandler;
import org.eclipse.jetty.servlet.ServletHolder;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;
import org.springframework.web.context.support.AnnotationConfigWebApplicationContext;
import org.springframework.web.servlet.DispatcherServlet;

public class Application {

    private static ApplicationContext appContext;
    public static void main(String[] args) throws Exception {
        appContext = new ClassPathXmlApplicationContext("application.xml");

        String[] beanDefinitionNames = appContext.getBeanDefinitionNames();
        for (String beanName : beanDefinitionNames) {
            System.out.println("beanName: " + beanName);
        }

        AnnotationConfigWebApplicationContext appWebContext = new AnnotationConfigWebApplicationContext();
        appWebContext.register(MemberController.class);
        appWebContext.register(ProductController.class);
        appWebContext.register(PurchaseController.class);
        appWebContext.register(SalesController.class);
        appWebContext.register(SourceController.class);
        appWebContext.register(StaffController.class);
        appWebContext.register(StatisticController.class);

        ServletHolder servletHolder = new ServletHolder(new DispatcherServlet(appWebContext));

        ServletContextHandler context = new ServletContextHandler();
        context.setContextPath("/");
        context.addServlet(servletHolder, "/*");

        Server server = new Server(8080);
        server.setHandler(context);

        server.start();
        server.join();
    }

    public static ApplicationContext getAppContext() {
        return appContext;
    }
}
