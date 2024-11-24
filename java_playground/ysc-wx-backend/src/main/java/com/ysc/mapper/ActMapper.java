package com.ysc.mapper;

import com.ysc.pojo.Act;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface ActMapper {

    @Select("select * from act")
    List<Act> selectAllActList();

    @Select("select * from act where act_id=#{actId}")
    Act selectActById(int actId);
}
