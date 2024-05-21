package me.iaksh.mapper;

import me.iaksh.entity.TotalIncome;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface TotalIncomeMapper {
    @Select("SELECT * FROM TotalIncome")
    List<TotalIncome> getIncome();
}
