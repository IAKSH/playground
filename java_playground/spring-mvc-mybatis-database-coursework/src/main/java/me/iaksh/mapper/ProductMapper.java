package me.iaksh.mapper;

import me.iaksh.entity.Product;
import org.apache.ibatis.annotations.*;

import java.util.List;

public interface ProductMapper {
    @Select("SELECT * FROM Product WHERE ProductID = #{id}")
    Product getById(@Param("id") long id);

    @Select("SELECT * FROM Product")
    List<Product> getAll();

    @Options(useGeneratedKeys = true, keyProperty = "productID", keyColumn = "ProductID")
    @Insert("INSERT INTO Product (Name, Brand, UnitPrice, Quantity) " +
            "VALUES (#{product.name}, #{product.brand}, #{product.unitPrice}, #{product.quantity})")
    void insert(@Param("product") Product product);

    @Update("UPDATE Product SET Name = #{product.name}, Brand = #{product.brand}, " +
            "UnitPrice = #{product.unitPrice}, Quantity = #{product.quantity} WHERE ProductID = #{product.productID}")
    void update(@Param("product") Product product);

    @Delete("DELETE FROM Product WHERE ProductID = #{id}")
    void deleteById(@Param("id") long id);
}