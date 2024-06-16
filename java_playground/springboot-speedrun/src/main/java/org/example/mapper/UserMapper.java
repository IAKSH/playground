package org.example.mapper;

import org.apache.ibatis.annotations.*;
import org.example.entity.User;
import org.example.entity.Role;

import java.util.List;

@Mapper
public interface UserMapper {
    @Select("SELECT * FROM roles WHERE id IN (SELECT role_id FROM user_roles WHERE user_id IN (SELECT id FROM users WHERE name=#{name}))")
    List<Role> getRolesByUserName(String name);

    @Select("SELECT * FROM users WHERE id = #{id}")
    User getUser(Long id);

    @Select("SELECT * FROM users WHERE name=#{name}")
    User getUserByName(String name);

    @Insert("INSERT INTO users (name, age, email, password) VALUES (#{name}, #{age}, #{email}, #{password})")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insertUser(User user);

    @Insert("INSERT INTO user_roles (user_id, role_id) VALUES (#{userId}, #{roleId})")
    int insertUserRole(@Param("userId") Long userId, @Param("roleId") Long roleId);

    @Update("UPDATE users SET name = #{name}, age=#{age}, email = #{email}, password = #{password} WHERE id = #{id}")
    int updateUser(User user);

    @Delete("DELETE FROM user_roles WHERE user_id = #{userId}")
    int deleteUserRole(@Param("userId") Long userId);

    @Delete("DELETE FROM users WHERE id = #{id}")
    int deleteUser(Long id);
}
