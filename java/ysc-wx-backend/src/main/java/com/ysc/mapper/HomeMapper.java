package com.ysc.mapper;

import com.ysc.pojo.HanFu;
import com.ysc.pojo.HanFuImage;
import com.ysc.pojo.Shop;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

@Mapper
public interface HomeMapper {

    @Select("select * from hanfu")
    List<HanFu> selectAll();

    @Select("select * from shop where shop_id=#{shopId}")
    Shop selectShopById(Integer shopId);

    @Select("select * from hanfu_image where hanfu_id=#{hanFuId};")
    List<HanFuImage> selectImageById(Integer hanFuId);

    @Select("select * from hanfu where hanfu_id=#{hanFuId}")
    HanFu selectHanFuInfoById(Integer hanFuId);

    @Update("update hanfu set views=#{views} where hanfu_id=#{hanFuId}")
    void upViews(int views, Integer hanFuId);

    @Select("select * from hanfu where name like concat('%',#{info},'%') or shop_name like concat('%',#{info},'%') or label like concat('%',#{info},'%') or price like concat('%',#{info},'%')")
    List<HanFu> selectAllByInfo(String info);

    @Select("select * from hanfu where label like concat('%',#{label},'%')")
    List<HanFu> selectHanFuListByLabel(String label);
}
