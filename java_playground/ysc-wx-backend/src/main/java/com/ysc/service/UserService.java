package com.ysc.service;

import com.ysc.pojo.User;
import org.springframework.stereotype.Service;

@Service
public interface UserService {

    void addUser(String account, String password, String telephone);

    Boolean userCheck(String account);

    User getUserInfo(String account);

    void changeName(Integer userId, String newNickname);

    void changeAvatarUrl(Integer userId, String newAvatarUrl);

    void changeGender(Integer userId, String gender);
}
