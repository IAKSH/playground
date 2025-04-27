package com.ysc.mapper;

import com.ysc.pojo.User;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.time.LocalDateTime;

@Mapper
public interface UserMapper {
    @Insert("insert into user (account,password,name,create_time,telephone) values (#{account},#{password},#{name},#{creatTime},#{telephone})")
    void insertUser(String account, String password, String name, LocalDateTime creatTime, String telephone);

    @Select("SELECT * FROM user WHERE account=#{account};")
    User selectUser(String account);

    @Update("update user set name=#{newNickname} where user_id=#{userId}")
    void updateNickname(Integer userId, String newNickname);

    @Update("update user set avatar=#{newAvatarUrl} where user_id=#{userId}")
    void updateAvatarUrl(Integer userId, String newAvatarUrl);

    @Update("update user set gender=#{gender} where user_id=#{userId}")
    void updateGender(Integer userId, String gender);
}
