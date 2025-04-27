package me.iaksh.mapper;

import me.iaksh.entity.DailySales;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface DailySalesMapper {
    @Select("SELECT * FROM DailySales")
    List<DailySales> get();
}
