package com.ysc.mapper;

import com.ysc.pojo.CloudCabinet;
import com.ysc.pojo.Shop;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface SiteMapper {

    @Select("select * from cloud_cabinet")
    List<CloudCabinet> selectTerminalAll();

    @Select("select * from shop")
    List<Shop> selectShopAll();

    @Select("select * from cloud_cabinet where cabinet_id=#{id}")
    CloudCabinet selectTerminalById(Integer id);
}
