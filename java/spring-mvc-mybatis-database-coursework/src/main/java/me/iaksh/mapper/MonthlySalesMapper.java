package me.iaksh.mapper;

import me.iaksh.entity.MonthlySales;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface MonthlySalesMapper {
    @Select("SELECT * FROM MonthlySales")
    List<MonthlySales> get();
}
