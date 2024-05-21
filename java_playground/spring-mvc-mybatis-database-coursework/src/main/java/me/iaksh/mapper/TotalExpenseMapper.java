package me.iaksh.mapper;

import me.iaksh.entity.TotalExpense;
import org.apache.ibatis.annotations.Select;

import java.util.List;

public interface TotalExpenseMapper {
    @Select("SELECT * FROM TotalExpense")
    List<TotalExpense> getExpense();
}
