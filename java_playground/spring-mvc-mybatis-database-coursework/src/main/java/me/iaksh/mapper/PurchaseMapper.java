package me.iaksh.mapper;

import me.iaksh.entity.Purchase;
import org.apache.ibatis.annotations.*;

import java.util.List;

public interface PurchaseMapper {
    @Select("SELECT * FROM Purchase WHERE PurchaseID = #{id}")
    Purchase getById(@Param("id") long id);

    @Select("SELECT * FROM Purchase")
    List<Purchase> getAll();

    @Options(useGeneratedKeys = true, keyProperty = "purchaseID", keyColumn = "PurchaseID")
    @Insert("INSERT INTO Purchase (ProductID, PurchaseTime, PurchaseUnitPrice, PurchaseQuantity, SourceID) " +
            "VALUES (#{purchase.productID}, #{purchase.purchaseTime}, #{purchase.purchaseUnitPrice}, #{purchase.purchaseQuantity}, #{purchase.sourceID})")
    void insert(@Param("purchase") Purchase purchase);

    @Update("UPDATE Purchase SET ProductID = #{purchase.productID}, PurchaseTime = #{purchase.purchaseTime}, " +
            "PurchaseUnitPrice = #{purchase.purchaseUnitPrice}, PurchaseQuantity = #{purchase.purchaseQuantity}, SourceID = #{purchase.sourceID} WHERE PurchaseID = #{purchase.purchaseID}")
    void update(@Param("purchase") Purchase purchase);

    @Delete("DELETE FROM Purchase WHERE PurchaseID = #{id}")
    void deleteById(@Param("id") long id);

    @Insert("{CALL PurchaseProduct(#{purchase.productID}, #{purchase.sourceID}, #{purchase.purchaseTime}, #{purchase.purchaseUnitPrice}, #{purchase.purchaseQuantity})}")
    void purchaseProduct(@Param("purchase") Purchase purchase);
}