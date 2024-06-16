# 前言

只能说 SpringBoot 还是走的 Spring 框架的 Bean 和 Spring MVC 的路子，充其量只是通过各种 Hack 自动化了一些配置，然后简化掉了 Spring 和 Spring MVC 原来一大堆的 xml 配置。

顺便把包管理和技术选型都给你包了，比较的无脑。

# 基本环境搭建

参考：[Spring Boot Gradle Plugin Reference Guide](https://docs.spring.io/spring-boot/docs/3.2.0-SNAPSHOT/gradle-plugin/reference/htmlsingle/#getting-started)

---

选用 OpenJDK 17，Gradle 以及 SpringBoot 提供的 Gradle 插件。

1. 在 `settings.gradle` 中引入 `spring` 的仓库
```gradle
pluginManagement {  
    repositories {  
        maven { url 'https://repo.spring.io/milestone' }  
        maven { url 'https://repo.spring.io/snapshot' }  
        gradlePluginPortal()  
    }  
}  
  
rootProject.name = 'springboot-speedrun'
```

2. 在 `build.gradle` 中引入 `org.springframework.boot` 插件以及相关依赖，还有仓库
```gradle
plugins {  
    id 'java'  
    id 'org.springframework.boot' version '3.2.0-SNAPSHOT'  
}  
  
apply plugin: 'io.spring.dependency-management'  
  
group = 'com.example'  
version = '0.0.1-SNAPSHOT'  
sourceCompatibility = '17'  
  
repositories {  
    mavenCentral()  
    maven { url 'https://repo.spring.io/milestone' }  
    maven { url 'https://repo.spring.io/snapshot' }  
}  
  
dependencies {  
    implementation 'org.springframework.boot:spring-boot-starter-web'  
}
```

3. 更新 Gradle 配置，现在你应该已经拥有 SpringBoot 了，可以尝试以下代码来测试
```java
package org.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@SpringBootApplication
public class MyApplication {

    private int i = 0;

    @RequestMapping("/")
    String home() {
        return "Hello World! " + (++i);
    }

    public static void main(String[] args) {
        SpringApplication.run(MyApplication.class, args);
    }

}
```

# Starter 包

参考：[使用Spring Boot进行开发 (springdoc.cn)](https://springdoc.cn/spring-boot/using.html#using)

---

>Starter 是一系列开箱即用的依赖，你可以在你的应用程序中导入它们。通过你 Starter，可以获得所有你需要的 Spring 和相关技术的一站式服务，免去了需要到处大量复制粘贴依赖的烦恼。例如，如果你想开始使用 Spring 和 JPA 进行数据库访问，那么可以直接在你的项目中导入 `spring-boot-starter-data-jpa` 依赖。

简单来说就是 SpringBoot 提供了一系列可能会用到的包，里面塞满了你可能会用的各种包。

既然你都用 SpringBoot 了，不如就直接用这些 Starter 算了。

# SpringApplication 类

参考：[核心特性 (springdoc.cn)](https://springdoc.cn/spring-boot/features.html#features)

---

SpringBoot 需要一个用来配置和启动整个程序的类，通过 SpringApplication 注解来标记这个类。在上面的例子中，我们直接使用一个 Controller 兼职 SpringApplication，但是 SpringApplication 也可以是独立的。

```java
package org.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {
    public static void main(String[] args) {
        System.out.println("Hello SpringBoot!");
        SpringApplication.run(Application.class, args);
    }
}
```

需要注意的就是这里用反射把自己这个类丢给了 SpringApplication，如果丢错了会找不到某些工厂 Bean

然后其他的 Controller 就还是照常写。

```java
package org.example;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HelloController {

    private int i = 0;

    @RequestMapping("/hello")
    String sayHello() {
        return "Hello World! " + (++i);
    }
}
```

感觉其实就是 Spring MVC

# RESTful API

这里实际上就还是 Spring MVC 的内容，SpringBoot 不过是简化了 Bean 还有依赖的配置。

```java
package org.example;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
public class HelloController {

    private int i = 0;

    @RequestMapping("/hello")
    String sayHello() {
        return "Hello World! " + (++i);
    }

    @RequestMapping("/hello/{name}")
    String sayHello(@PathVariable String name) {
        return "Hello " + name + "!";
    }

    @GetMapping(value = "/json/{id}",
            produces = "application/json;charset=UTF-8")
    public String testHttpGet(@PathVariable Long id) throws JsonProcessingException {
        Map<String,Object> map=new HashMap<>();
        map.put("name","aihao");
        map.put("age",23);
        map.put("gender","男");
        map.put("received_id",id);
        ObjectMapper mapper = new ObjectMapper();
        return mapper.writeValueAsString(map);
    }

    @PostMapping(value = "/json/post",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String testHttpPost(@RequestBody String strStaff) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(strStaff, Map.class).toString();
    }

    @PutMapping(value = "/json/put",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String testHttpPut(@RequestBody String strStaff) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(strStaff, Map.class).toString();
    }

    @DeleteMapping(value = "/json/delete/{id}",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String testHttpDelete(@RequestBody String strStaff) throws JsonProcessingException {
        ObjectMapper mapper = new ObjectMapper();
        return mapper.readValue(strStaff, Map.class).toString();
    }
}
```

# 配置

参考：[核心特性 (springdoc.cn)](https://springdoc.cn/spring-boot/features.html#features.external-config)

---

简单地说，Spring 会维护一个 `Environment`，你可以在这里写入你的配置信息。这个 `Environment` 可以从命令行以及外部配置文件读入，然后在 Java 代码中使用类似于下述代码进行使用。

```java
@Component
public class MyBean {
    @Value("${name}") private String name;
    // ...
}
```

## 从命令行配置

SpringApplication 会将任何命令行长参数（以 `--` 开头的）参数转化为 ` property ` 然后塞入 Spring 的 ` Environment `，也就是说，对于上面的例子，你甚至可以直接这样对其进行配置：

```sh
java -jar app.jar --name="Spring"
```

命令行的属性配置总是优先于基于文件的，但是命令行配置可以被手动关闭：

```java
package org.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication app = new SpringApplication(Application.class);
        app.setAddCommandLineProperties(false);
        app.run(args);
    }
}
```
## 外部配置文件 `application.properties/yaml `


>当你的应用程序启动时，Spring Boot 会自动从以下位置找到并加载 `application.properties` 和 `application.yaml` 文件。
>
>1. Classpath
    >	1. Classpath 根路径
>	2. Classpath 下的 `/config` 包
>2. 当前目录
    >	1. 当前目录下
>	2. 当前目录下的 `config/` 子目录
>	3. `config/` 子目录的直接子目录
>
>列表按优先级排序（较低项目的值覆盖较早项目的值）。加载的文件被作为 `PropertySources` 添加到 Spring 的 `Environment` 中。

还可以从命令行参数写入 `Environment` 来让 Spring 使用其他名字的配置文件：

```sh
java -jar myproject.jar --spring.config.name=myproject
```

或者使用 `spring.config.location` 来引用一个或者多个明确的位置：

```sh
java -jar myproject.jar --spring.config.location=\ optional:classpath:/default.properties,\ optional:classpath:/override.properties
```

其中的 `optional:前缀` 表示该配置文件是可选的，可以不存在。

# Profiles

看不太懂，暂略

# 日志

懒得写，略

# 国际化

懒得写，略

# JSON

SpringBoot 提供与 `Gson`，`Jackson`，`JSON-B` 这些 JSON 库的集成，但默认首选 `Jackson`。

对于单纯 `Jackson`，你通常需要自己编写序列化和反序列化类，然后向 `Jackson` 注册，但是 Springboot 提供了一个更简单的 ` @JsonComponent ` 注解。

```java
import java.io.IOException;

import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.JsonSerializer;
import com.fasterxml.jackson.databind.SerializerProvider;

import org.springframework.boot.jackson.JsonComponent;

@JsonComponent
public class MyJsonComponent {

    public static class Serializer extends JsonSerializer<MyObject> {

        @Override
        public void serialize(MyObject value, JsonGenerator jgen, SerializerProvider serializers) throws IOException {
            jgen.writeStartObject();
            jgen.writeStringField("name", value.getName());
            jgen.writeNumberField("age", value.getAge());
            jgen.writeEndObject();
        }
    }

    public static class Deserializer extends JsonDeserializer<MyObject> {
    
        @Override
        public MyObject deserialize(JsonParser jsonParser, DeserializationContext ctxt) throws IOException {
            ObjectCodec codec = jsonParser.getCodec();
            JsonNode tree = codec.readTree(jsonParser);
            String name = tree.get("name").textValue();
            int age = tree.get("age").intValue();
            return new MyObject(name, age);
        }
    }
}
```

> `ApplicationContext` 中的所有 `@JsonComponent` Bean 都会自动向 Jackson 注册。因为 `@JsonComponent` 是用 `@Component` 元注解的，所以通常的组件扫描规则适用。

# 数据库连接（PostgreSQL）

1. 导入数据库驱动的依赖（还有 JPA ?）
```gradle
dependencies {
	// others...
	implementation 'org.postgresql:postgresql'
	implementation 'org.springframework.boot:spring-boot-starter-data-jpa'
	// ...
}
```

2. 由于 springboot 给你什么都包完了，所以只需要改改 `environment`，就能让 springboot 帮你连好数据库
```properties
spring.datasource.url=jdbc:postgresql://localhost:5432/test  
spring.datasource.username=root  
spring.datasource.password=root
```

3. 然后数据库就能连上了，可以直接用了，比如直接拿 spring 的 JDBCTemplate 试试读
```java
@GetMapping("/without-orm")  
public ResponseEntity<List<Map<String,Object>>> getAllUsersWithoutORM() {  
    String sql = "SELECT * FROM users";  
    List<Map<String,Object>> res = jdbcTemplate.queryForList(sql);  
    return ResponseEntity.ok(res);  
}
```

# `mybatis-spring` 集成

除了上面那种用 Spring 提供的 `JdbcTemplate` 直接朝数据库跑 SQL 的外，还可以用 ORM 框架。虽然 Mybatis 不完全是。

Springboot 官方似乎没有 Mybatis 支持，但是 `mybatis-spring` 提供了一个 starter

```gradle
dependencies {
	// others...
	implementation 'org.mybatis.spring.boot:mybatis-spring-boot-starter:3.0.3'
	// ...
}
```

Mybatis 还是老用法，写 Entity 和 Mapper，然后写 Mapper 的 xml，最后写 Service 和 Controller。springboot 简化了原来通过 xml 配置 bean 的过程，其他的还是一样。

如果是用 Mybatis-plus，或者其他的几个真正的 ORM 框架，是可以做到无 xml 配置的。

1. 编写 Entity 类，这里偷了懒，用 `lombok` 自动生成 getter 和 setter 函数。
```java
package org.example.entity;  
  
import lombok.Data;  
  
@Data  
public class User {  
    private Long id;  
    private String name;  
    private Integer age;  
    private String email;  
}
```

2. 编写 Mapper 类
```java
package org.example.mapper;  
  
import org.apache.ibatis.annotations.Mapper;  
import org.example.entity.User;  
  
@Mapper  
public interface UserMapper {  
    User getUser(Long id);  
    int insertUser(User user);  
    int updateUser(User user);  
    int deleteUser(Long id);  
}
```

3. 在 `Environment` 中添加关于 mapper xml 配置的部分，主要是指名 entity 包名以及 mapper 的 xml 的位置。
```properties
spring.jpa.hibernate.ddl-auto=update  
mybatis.type-aliases-package=org.example.entity  
mybatis.mapper-locations=classpath:mapper/*.xml
```

4. 为每个 mapper 编写 xml，在其中绑定 SQL。*（没记错的话，之前 Spring MVC 的时候找到过可以全程注解的无 xml 方案，之后再弄吧）*
```xml
<?xml version="1.0" encoding="UTF-8" ?>  
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">  
  
<mapper namespace="org.example.mapper.UserMapper">  
    <resultMap id="UserResultMap" type="org.example.entity.User">  
        <id property="id" column="id" />  
        <result property="name" column="name" />  
        <result property="age" column="age" />  
        <result property="email" column="email" />  
    </resultMap>  
    <select id="getUser" resultMap="UserResultMap">  
        SELECT * FROM users WHERE id = #{id}  
    </select>  
  
    <insert id="insertUser">  
        INSERT INTO users (name, age, email) VALUES (#{name}, #{age}, #{email})  
    </insert>  
  
    <update id="updateUser">  
        UPDATE users SET name = #{name}, age=#{age}, email = #{email} WHERE id = #{id}  
    </update>  
  
    <delete id="deleteUser">  
        DELETE FROM users WHERE id = #{id}  
    </delete>  
</mapper>
```

5. 编写对应 Service 类
```java
package org.example.service;  
  
import org.example.entity.User;  
import org.example.mapper.UserMapper;  
import org.springframework.beans.factory.annotation.Autowired;  
import org.springframework.stereotype.Service;  
  
@Service  
public class UserService {  
    @Autowired  
    private UserMapper userMapper;  
  
    public User getUser(Long id) {  
        return userMapper.getUser(id);  
    }  
  
    public int insertUser(User user) {  
        return userMapper.insertUser(user);  
    }  
  
    public int updateUser(User user) {  
        return userMapper.updateUser(user);  
    }  
  
    public int deleteUser(Long id) {  
        return userMapper.deleteUser(id);  
    }  
}
```

6. 编写对应 Controller 类
```java
package org.example.controller;  
  
import org.example.entity.User;  
import org.example.service.UserService;  
import org.springframework.beans.factory.annotation.Autowired;  
import org.springframework.http.ResponseEntity;  
import org.springframework.jdbc.core.JdbcTemplate;  
import org.springframework.web.bind.annotation.*;  
  
import java.util.List;  
import java.util.Map;  
  
@RestController  
@RequestMapping("/users")  
public class UserController {  
  
    private final UserService userService;  
  
    @Autowired  
    public UserController(UserService userService) {  
        this.userService = userService;  
    }  
  
    @Autowired  
    private JdbcTemplate jdbcTemplate;  
  
    // 返回的 ResponseEntity<User> 将会被自动转换为 JSON（默认使用Jackson库）  
    @PostMapping  
    public ResponseEntity<User> createUser(@RequestBody User user) {  
        userService.insertUser(user);  
        return ResponseEntity.ok(user);  
    }  
  
    @GetMapping("/{id}")  
    public ResponseEntity<User> getUserById(@PathVariable Long id) {  
        User user = userService.getUser(id);  
  
        if (user != null) {  
            return ResponseEntity.ok(user);  
        } else {  
            return ResponseEntity.notFound().build();  
        }  
    }  
  
    @GetMapping("/without-orm")  
    public ResponseEntity<List<Map<String,Object>>> getAllUsersWithoutORM() {  
        String sql = "SELECT * FROM users";  
        List<Map<String,Object>> res = jdbcTemplate.queryForList(sql);  
        return ResponseEntity.ok(res);  
    }  
}
```

## 纯注解 Mybatis

只需要删除上面提到的 `application.properties` 中关于 mybatis 的内容，然后把 mapper xml 删了，SQL 搬到 mapper 类的方法注解上就是了

```java
package org.example.mapper;  
  
import org.apache.ibatis.annotations.*;  
import org.example.entity.User;  
  
@Mapper  
public interface UserMapper {  
    @Select("SELECT * FROM users WHERE id = #{id}")  
    User getUser(Long id);  
  
    @Insert("INSERT INTO users (name, age, email) VALUES (#{name}, #{age}, #{email})")  
    int insertUser(User user);  
  
    @Update("UPDATE users SET name = #{name}, age=#{age}, email = #{email} WHERE id = #{id}")  
    int updateUser(User user);  
  
    @Delete("DELETE FROM users WHERE id = #{id}")  
    int deleteUser(Long id);  
}
```

十分的简单，十分的易用。

## 版本兼容性问题

由于 Java 程序员特有的疯狂反射以及 Java 框架特有的四处 hack，`mybatis-spring` 对你使用的 `spring` 的版本非常的敏感，已知 `spring 3.2.0` 在使用 `mybatis-spring 3.0.3` 时正常，如果低于 `mybatis-spring 3.0.2` 则十分可能出现只有运行时才会显现出来的某种神必 bug

# Spring Security 集成

`spring-security` 主要解决了**认证**和**授权**的问题，简单来说就是验证一个用户是否有访问权限，以及给予访问权限。

`spring-security` 的一个典型应用就是用户登录，当然也可以用来做其他各种各样的东西。

`spring-security` 在 springboot 中是无缝集成的，你不需要为了加入 `spring-security` 修改已有的（大部分）代码，只需要添加一些新的配置类，以及对应的与数据库的交互设施即可。

## JWT

参考：
+  [Cookie、Session、Token、JWT一次性讲完_哔哩哔哩_bilibili](https://www.bilibili.com/video/BV18u4m1K7D4/?spm_id_from=333.337.search-card.all.click&vd_source=aba245c5d1c4487c2355023d2870a6f7)
+ [【全网最细致】SpringBoot整合Spring Security + JWT实现用户认证_spring boot+securrtiy+jwt实现用户认证流程-CSDN博客](https://blog.csdn.net/qq_44709990/article/details/123082560)
+ [2023-10 最新jsonwebtoken-jjwt 0.12.3 基本使用-CSDN博客](https://blog.csdn.net/qq_50969362/article/details/134100542)
+ [Spring Security：升级已弃用的 WebSecurityConfigurerAdapter - spring 中文网 (springdoc.cn)](https://springdoc.cn/spring-deprecated-websecurityconfigureradapter/)
+ [SpringBoot3 - Spring Security 6.0 Migration_authorizerequests()' is deprecated-CSDN博客](https://blog.csdn.net/nyzzht123/article/details/129794744)
+ [【Day - 23】Spring Security 6.1.x：實現JWT身份驗證 (中) - iT 邦幫忙::一起幫忙解決難題，拯救 IT 人的一天 (ithome.com.tw)](https://ithelp.ithome.com.tw/articles/10333422)
+ [Ons-diweni/Spring-Security-6-JWT: Spring Security 6 demo : Spring Boot application that makes use of JWT for securing an exposed REST API : Spring boot 3.0.2 - Java 17 - Spring Security 6.0.1 (github.com)](https://github.com/Ons-diweni/Spring-Security-6-JWT)

---

JWT 是 Token 的一种标准，使用似乎比较广泛，简单地说就是前端登录后服务器生成一个 Token 给前端，然后前端之后每次请求 API 都得在 http 头部加这个 JWT，然后 http 报文还是放要传的数据，就能完成鉴权+传递数据了。

JWT 的生存周期完全由服务器决定，但是 JWT 通常是不加密的，而且前端将自己管理这个 JWT。

~~而且 JWT 因为无状态（http 也是无状态的），所以 JWT 一旦发放就不能修改。~~

虽然 JWT 默认不加密，但是其带有（哈希）校验段，所以 JWT 是无法被篡改的（虽然有可能被盗用）。

相当于就是一块谁都能读出来上面的信息，也可能被人偷走，但是无法被修改的令牌。

`spring-security` 中似乎还是比较容易实现 JWT 的。

## 使用 spring-security 实现基于 JWT 的身份验证

这里版本问题带来的坑非常多，首先是确定版本

```gradle
dependencies {  
    implementation 'org.springframework.boot:spring-boot-starter-security:3.2.3'  
    implementation 'org.springframework.boot:spring-boot-starter-web:3.2.3'  
    implementation 'org.mybatis.spring.boot:mybatis-spring-boot-starter:3.0.3'  
    implementation 'org.postgresql:postgresql'  
    implementation 'org.springframework.boot:spring-boot-starter-data-jpa:3.2.3'  
    testImplementation 'org.springframework.boot:spring-boot-starter-test:3.2.3'  
    implementation 'io.jsonwebtoken:jjwt:0.12.5'  
    compileOnly 'org.projectlombok:lombok'  
    annotationProcessor 'org.projectlombok:lombok'  
    //developmentOnly("org.springframework.boot:spring-boot-docker-compose")  
    testImplementation 'org.springframework.security:spring-security-test:3.2.3'  
}
```

然后就是写 `SecurityConfig` 来配置哪些地方需要什么权限，以及创建一些 bean，比如密码解码器；写 `AuthenticationFilter` 来拦截流量进行验证；最后就是 `JwtService` 和对应 Controller，Mapper，Entity 来实现整个 JWT 分发和验证。

具体见下
[a basically usable JWT using spring-security · IAKSH/playground@da1e878 (github.com)](https://github.com/IAKSH/playground/commit/da1e8782b4005491e1a2aae05dfcbbe4c99f4662)

以及基于 BCrypt 的加密密码登录
[encrypted password · IAKSH/playground@370aaef (github.com)](https://github.com/IAKSH/playground/commit/370aaef073fcb7483b06f90dfd005efb3794131c)

另外，关于前端应该传输明文密码还是密文，大多数场景下都可以把这个问题丢给 https 解决，然后前端直接在报文里赛明文密码。

然后是带角色（role）的权限管理

[role (not in JWT) · IAKSH/playground@692dcc7 (github.com)](https://github.com/IAKSH/playground/commit/692dcc7bb2df5852cca996aec436267acb60394d)

以及直接使用请求的 JWT 中的 role，而不是重新从数据库查询

[role (in JWT) · IAKSH/playground@a198c31 (github.com)](https://github.com/IAKSH/playground/commit/a198c31ecdfcdd498985761f7232839b1ead82f5)

## 使用 Redis 实现 JWT 登出

参考：
+  [【Day - 24】Spring Security 6.1.x JWT身份驗證 (下)：透過Redis實作登出功能 - iT 邦幫忙::一起幫忙解決難題，拯救 IT 人的一天 (ithome.com.tw)](https://ithelp.ithome.com.tw/articles/10334160)
+ [在 Spring Boot 中整合、使用 Redis - spring 中文网 (springdoc.cn)](https://springdoc.cn/spring-boot-data-redis/)

---

由于 JWT 是无状态的，而且发出后就不能修改，有时候想要提前登出，令 JWT 无效，就得上一些特殊手段了。

原理很简单，就是维护一个 JWT 黑名单 （通常丢给 redis），每次收到请求，验证 JWT 的时候拿去比对以下，如果在黑名单里就无效。

[JWT logout/block using Redis · IAKSH/playground@a28d109 (github.com)](https://github.com/IAKSH/playground/commit/a28d1096f23b3a7e264dc4288828edda5f69dc4f)

另外这个 JWT 黑名单还得自动删除过期的 JWT，只需要封禁尚未过期的就行。

[JWT block record in Redis delete when JWT is outdated · IAKSH/playground@46a74e1 (github.com)](https://github.com/IAKSH/playground/commit/46a74e128f033905979cfee0729742e4fbf37da6)

# Websocket 集成

参考：
+  [Spring Boot 集成 WebSocket（原生注解与Spring封装）_springboot集成websocket-CSDN博客](https://blog.csdn.net/qq991658923/article/details/127022522)
+ [在 Spring Boot 中整合、使用 WebSocket - spring 中文网 (springdoc.cn)](https://springdoc.cn/spring-boot-websocket/)

---

实际上疑似有点过于简单了，由于 websocket 本来就是再 jakarta 里包了一遍的，spring 也有给相应的 starter，所以编写和使用起来异常无脑

[websocket · IAKSH/playground@45aa93b (github.com)](https://github.com/IAKSH/playground/commit/45aa93baafa1baf0cc9feb82adbba3927294c202)

# 测试 TODO