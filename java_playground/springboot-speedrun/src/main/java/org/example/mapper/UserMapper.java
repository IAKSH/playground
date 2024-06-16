package org.example.mapper;

import org.apache.ibatis.annotations.*;
import org.example.entity.User;

@Mapper
public interface UserMapper {
    @Select("SELECT * FROM users WHERE id = #{id}")
    User getUser(Long id);

    @Select("SELECT * FROM users WHERE name=#{name}")
    User getUserByName(String name);

    @Insert("INSERT INTO users (name, age, email, password) VALUES (#{name}, #{age}, #{email}, #{password})")
    int insertUser(User user);

    @Update("UPDATE users SET name = #{name}, age=#{age}, email = #{email}, password = #{password} WHERE id = #{id}")
    int updateUser(User user);

    @Delete("DELETE FROM users WHERE id = #{id}")
    int deleteUser(Long id);
}
