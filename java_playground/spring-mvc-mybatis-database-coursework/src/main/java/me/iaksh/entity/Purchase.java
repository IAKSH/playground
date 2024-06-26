package me.iaksh.entity;

import java.math.BigDecimal;
import java.sql.Timestamp;

public class Purchase {
    private Integer purchaseID;
    private Integer productID;
    private Timestamp purchaseTime;
    private BigDecimal purchaseUnitPrice;
    private Integer purchaseQuantity;
    private Integer sourceID;

    public Integer getProductID() {
        return productID;
    }

    public Integer getSourceID() {
        return sourceID;
    }

    public BigDecimal getPurchaseUnitPrice() {
        return purchaseUnitPrice;
    }

    public Timestamp getPurchaseTime() {
        return purchaseTime;
    }

    public Integer getPurchaseID() {
        return purchaseID;
    }

    public Integer getPurchaseQuantity() {
        return purchaseQuantity;
    }

    public void setProductID(Integer productID) {
        this.productID = productID;
    }

    public void setSourceID(Integer sourceID) {
        this.sourceID = sourceID;
    }

    public void setPurchaseID(Integer purchaseID) {
        this.purchaseID = purchaseID;
    }

    public void setPurchaseQuantity(Integer purchaseQuantity) {
        this.purchaseQuantity = purchaseQuantity;
    }

    public void setPurchaseTime(Timestamp purchaseTime) {
        this.purchaseTime = purchaseTime;
    }

    public void setPurchaseUnitPrice(BigDecimal purchaseUnitPrice) {
        this.purchaseUnitPrice = purchaseUnitPrice;
    }
}
