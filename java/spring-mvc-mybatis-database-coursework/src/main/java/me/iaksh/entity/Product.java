package me.iaksh.entity;

import java.math.BigDecimal;

public class Product {
    private Integer productID;
    private String name;
    private String brand;
    private BigDecimal unitPrice;
    private Integer quantity;

    public String getName() {
        return name;
    }

    public BigDecimal getUnitPrice() {
        return unitPrice;
    }

    public Integer getProductID() {
        return productID;
    }

    public Integer getQuantity() {
        return quantity;
    }

    public String getBrand() {
        return brand;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setProductID(Integer productID) {
        this.productID = productID;
    }

    public void setBrand(String brand) {
        this.brand = brand;
    }

    public void setQuantity(Integer quantity) {
        this.quantity = quantity;
    }

    public void setUnitPrice(BigDecimal unitPrice) {
        this.unitPrice = unitPrice;
    }
}
