package me.iaksh.mapper;

import me.iaksh.entity.Sales;
import org.apache.ibatis.annotations.*;

import java.util.List;

public interface SalesMapper {
    @Select("SELECT * FROM Sales WHERE SalesID = #{id}")
    Sales getById(@Param("id") long id);

    @Select("SELECT * FROM Sales")
    List<Sales> getAll();

    @Options(useGeneratedKeys = true, keyProperty = "salesID", keyColumn = "SalesID")
    @Insert("INSERT INTO Sales (ProductID, SaleTime, ActualUnitPrice, SoldQuantity, MemberID) " +
            "VALUES (#{sales.productID}, #{sales.saleTime}, #{sales.actualUnitPrice}, #{sales.soldQuantity}, #{sales.memberID})")
    void insert(@Param("sales") Sales sales);

    @Update("UPDATE Sales SET ProductID = #{sales.productID}, SaleTime = #{sales.saleTime}, " +
            "ActualUnitPrice = #{sales.actualUnitPrice}, SoldQuantity = #{sales.soldQuantity}, MemberID = #{sales.memberID} WHERE SalesID = #{sales.salesID}")
    void update(@Param("sales") Sales sales);

    @Delete("DELETE FROM Sales WHERE SalesID = #{id}")
    void deleteById(@Param("id") long id);
}