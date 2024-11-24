package com.ysc.service.impl;

import com.ysc.mapper.UserMapper;
import com.ysc.pojo.User;
import com.ysc.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.ArrayList;

import org.springframework.security.core.userdetails.UserDetailsService;

@Service
public class UserServiceImpl implements UserService,UserDetailsService {

    @Autowired
    private UserMapper userMapper;

    @Override
    public UserDetails loadUserByUsername(String username) throws UsernameNotFoundException {
        User user = userMapper.selectUser(username);
        if (user == null) {
            throw new UsernameNotFoundException("Could not find user");
        }
        // 两个参数分别是用户名和密码
        return new org.springframework.security.core.userdetails.User(user.getAccount(), user.getPassword(), new ArrayList<>());
    }

    @Override
    public void addUser(String account, String password, String telephone) {
        String name="微信用户_"+account;
        LocalDateTime creatTime=LocalDateTime.now();

//        System.out.println(creatTime);
        userMapper.insertUser(account,password,name,creatTime,telephone);
    }

    @Override
    public Boolean userCheck(String account) {
        return userMapper.selectUser(account) != null;
    }

    @Override
    public User getUserInfo(String account) {
        return userMapper.selectUser(account);
    }

    @Override
    public void changeName(Integer userId, String newNickname) {
        userMapper.updateNickname(userId,newNickname);
    }

    @Override
    public void changeAvatarUrl(Integer userId, String newAvatarUrl) {
        userMapper.updateAvatarUrl(userId,newAvatarUrl);
    }

    @Override
    public void changeGender(Integer userId, String gender) {
        userMapper.updateGender(userId,gender);
    }
}
