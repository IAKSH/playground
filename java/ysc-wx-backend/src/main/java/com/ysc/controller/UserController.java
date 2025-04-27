package com.ysc.controller;


import com.ysc.pojo.Result;
import com.ysc.pojo.User;
import com.ysc.security.JwtService;
import com.ysc.service.UserService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/user")
@Slf4j
public class UserController {
    @Autowired
    private UserService myService;
    @Autowired
    private AuthenticationManager authenticationManager;
    @Autowired
    private UserDetailsService userDetailsService;
    @Autowired
    private JwtService jwtService;

    @PostMapping("/login")
    public Result userLogin(@RequestBody Map<String, String> loginData) {
        String account = loginData.get("account");
        String password = loginData.get("password");
        try {
            authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(account, password)
            );
        } catch (Exception e) {
            e.printStackTrace();
            return Result.error("Incorrect username or password");
        }
        final UserDetails userDetails = userDetailsService.loadUserByUsername(account);
        final String jwt = jwtService.generateToken(userDetails);
        return Result.success(jwt);
    }

    @PostMapping("/logout")
    public Result userLogout(@RequestHeader("Authorization") String authHeader) {
        if(authHeader != null && authHeader.startsWith("Bearer ")) {
            // 搓索引7开始，获取"Bearer "后面的Token
            String jwt = authHeader.substring(7);
            jwtService.addTokenToBlacklist(jwt);
        }
        return Result.success();
    }

    @GetMapping("/check")
    public Result checkIsRegister(@RequestBody String account){
        String isR= String.valueOf(myService.userCheck(account));
        return Result.success(isR);
    }

    @PostMapping("/register")
    public Result userRegister(@RequestBody User user) {
        String account = user.getAccount();
        String password = user.getPassword();
        String telephone = user.getTelephone();

        PasswordEncoder passwordEncoder = new BCryptPasswordEncoder();
        String encodedPassword = passwordEncoder.encode(password);

        myService.addUser(account, encodedPassword, telephone);
        return Result.success();
    }

    @PostMapping("/nickname/{userId}")
    public Result modifyNickname(@PathVariable Integer userId, String newNickname){
        myService.changeName(userId,newNickname);
        return Result.success();
    }

    @PostMapping("/avatar/{userId}")
    public Result modifyAvatar(@PathVariable Integer userId ,String newAvatarUrl){
        myService.changeAvatarUrl(userId,newAvatarUrl);
        return Result.success();
    }

    @PostMapping("/gender/{userId}")
    public Result modifyGender(@PathVariable Integer userId,String gender){
        myService.changeGender(userId,gender);
        return Result.success();
    }
}
