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


