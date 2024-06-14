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