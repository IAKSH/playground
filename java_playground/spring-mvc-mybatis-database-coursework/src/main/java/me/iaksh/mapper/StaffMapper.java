package me.iaksh.mapper;

import me.iaksh.entity.Staff;
import org.apache.ibatis.annotations.*;

import java.util.List;

public interface StaffMapper {
    @Select("SELECT * FROM Staff WHERE StaffID = #{id}")
    Staff getById(@Param("id") long id);

    @Select("SELECT * FROM Staff")
    List<Staff> getAll();

    @Options(useGeneratedKeys = true, keyProperty = "staffID", keyColumn = "StaffID")
    @Insert("INSERT INTO Staff (Name, Gender, Age, MonthlySalary) " +
            "VALUES (#{staff.name}, #{staff.gender}, #{staff.age}, #{staff.monthlySalary})")
    void insert(@Param("staff") Staff staff);

    @Update("UPDATE Staff SET Name = #{staff.name}, Gender = #{staff.gender}, " +
            "Age = #{staff.age}, MonthlySalary = #{staff.monthlySalary} WHERE StaffID = #{staff.staffID}")
    void update(@Param("staff") Staff staff);

    @Delete("DELETE FROM Staff WHERE StaffID = #{id}")
    void deleteById(@Param("id") long id);
}
