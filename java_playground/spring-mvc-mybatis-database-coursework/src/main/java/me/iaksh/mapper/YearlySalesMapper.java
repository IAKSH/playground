package me.iaksh.mapper;

import me.iaksh.entity.YearlySales;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface YearlySalesMapper {
    @Select("SELECT * FROM YearlySales")
    List<YearlySales> get();
}
